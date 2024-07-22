#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm
import xarray as xr
import os
import scipy as sp
import sys
# I want to import lib.py from src folder in top level directory
top_dir = '../../../'
sys.path.append(top_dir + 'src')
import lib


def sim_connectome(n, syn_count_sgn, post_root_id_unique, pre_root_id_unique, scale_var=1):
    """
    Simulates a connectome from a normal distribution with mean = syn_count_sgn and var=|sync_count_sgn|*scale_var 
    then scales it to be stable for a dynamics matrix.

    Parameters:
    n (int): Number of neurons.
    syn_count_sgn (numpy.ndarray): Vector of synaptic counts multiplied by the inferred sign.
    post_root_id_unique (numpy.ndarray): Array of post-synaptic indices (rows).
    pre_root_id_unique (numpy.ndarray): Array of pre-synaptic indices (columns).
    scale_var (float, optional): Scaling factor for the variance of the normal distribution. Default is 1.

    Returns:
    W_sim (scipy.sparse.csr_matrix): Simulated dynamics matrix.
    """
    syn_count_sgn_sim = np.random.normal(loc=syn_count_sgn, 
                            scale=np.abs(syn_count_sgn*scale_var)**0.5, 
                            size=syn_count_sgn.shape)#drawn from normal distribution

    C_sim = csr_matrix((syn_count_sgn_sim, (post_root_id_unique, pre_root_id_unique)), 
                            shape=(n, n), dtype='float64')#transform into sparse matrix
    eigenvalues, _ = eigs(C_sim, k=1)#just get largest eigenvalue
    scale_sim = 1/np.abs(eigenvalues[0])
    W_sim = C_sim*scale_sim  # scale connectome by largest eigenvalue so that it is stable
    return W_sim


seed = 1
data_dir = top_dir + 'data/'
df_sgn = pd.read_csv(data_dir + 'connectome_sgn_cnt_prob.csv', index_col=0)
syn_count_sgn = df_sgn['syn_count_sgn'].values
post_root_id_unique = df_sgn['post_root_id_ind'].values
pre_root_id_unique = df_sgn['pre_root_id_ind'].values
connectome_scaled = sp.sparse.load_npz(data_dir + 'connectome_sgn_cnt_scaled_sp_mat.npz')

source_neuron = np.array([68521])#chosen because it has many downstream neurons
n_neurons = connectome_scaled.shape[0]

n_sims = 10
a_C = connectome_scaled[:, source_neuron].toarray()

#T_subs = list(np.arange(5, 50, 5)) + list(np.arange(50, 100, 10)) + list(np.arange(100, 1000, 100)) +  list(np.arange(1000, 11000, 1000))
T_subs = [10, 50,  100, 500, 1000,  5000, 10000]
#data array for suff stats of target neurons
dims = ['w', 'sim', 'T_sub', 'est']
coords = {'w':range(len(a_C)), 'sim':range(n_sims), 'T_sub':T_subs, 'est':['XTY', 'sig2', 'IV']}
da_target = xr.DataArray(np.zeros((len(a_C), n_sims, len(T_subs), 3)), dims=dims, coords=coords)

#data array for suff stats of source neurons
dims = ['sim', 'T_sub']
coords = {'sim':range(n_sims), 'T_sub':T_subs}
da_source = xr.DataArray(np.zeros((n_sims, len(T_subs))), dims=dims, coords=coords)

#data array for true weights (from simulated connectomes)
dims = ['w', 'sim']
coords = {'w':range(len(a_C)), 'sim':range(n_sims)}
w_true = xr.DataArray(np.zeros((len(a_C), n_sims)), dims=dims, coords=coords)

for sim in tqdm(range(n_sims)):
    W = sim_connectome(n_neurons, syn_count_sgn, post_root_id_unique, pre_root_id_unique)
    R, L = lib.simulate_sparse_VAR1(W, source_neuron, T=np.max(T_subs), seed=sim, laser_power=10., noise_var=1., )
    w_true[:, sim] = W[:, source_neuron].toarray().squeeze()
    for T_sub in tqdm(T_subs):
        X = R[source_neuron, :T_sub]
        Y = R[:, :T_sub]
        a_L = L[:, :T_sub]
        XTX, XTY, sig2, hat_W_xy_IV = lib.suff_stats_fit_prior(X, Y, a_L)
        da_target.loc[dict(sim=sim, T_sub=T_sub, est='XTY')] = XTY.squeeze()
        da_target.loc[dict(sim=sim, T_sub=T_sub, est='sig2')] = sig2.squeeze()
        da_target.loc[dict(sim=sim, T_sub=T_sub, est='IV')] = hat_W_xy_IV.squeeze()
        da_source.loc[dict(sim=sim, T_sub=T_sub)] = XTX.squeeze()
ds = xr.Dataset({'target':da_target, 'source':da_source, 'w_true':w_true})

# %%
#data array for estimates
dims = ['w', 'sim', 'T_sub', 'est']
coords = {'w':range(len(a_C)), 'sim':range(n_sims), 'T_sub':T_subs, 'est':['bayes', 'IV']}
da_est = xr.DataArray(np.zeros((len(a_C), n_sims, len(T_subs), 2)), dims=dims, coords=coords)
#now use suff stats to get estimate
for sim in tqdm(range(n_sims)):
    for T_sub in (T_subs):
        XTX = ds['source'].sel(sim=sim, T_sub=T_sub).values[None,None]
        XTY = ds['target'].sel(sim=sim, T_sub=T_sub, est='XTY').values[None, :]
        sig2 = ds['target'].sel(sim=sim, T_sub=T_sub, est='sig2').values
        hat_W_xy_IV_bayes = lib.fit_prior_w_suff_stat(XTX, XTY, sig2, a_C)
        da_est.loc[dict(sim=sim, T_sub=T_sub, est='bayes')] = hat_W_xy_IV_bayes.squeeze()
        da_est.loc[dict(sim=sim, T_sub=T_sub, est='IV')] = ds['target'].sel(sim=sim, T_sub=T_sub, est='IV').values.squeeze()
#%%
#save all results as netcdf files
#if these files already exist, delete them
if os.path.exists(data_dir + '/sim_results.nc'):
    os.remove(data_dir + '/sim_results.nc')
if os.path.exists(data_dir + '/sim_estimates.nc'):
    os.remove(data_dir + '/sim_estimates.nc')
ds.to_netcdf(data_dir + '/sim_results.nc')
da_est.to_netcdf(data_dir + '../data/sim_estimates.nc')


