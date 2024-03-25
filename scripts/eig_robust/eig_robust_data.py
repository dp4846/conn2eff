#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import xarray as xr
from tqdm import tqdm
np.random.seed(42)

# this generates several different eigendecompositions of the connectome matrix transformed in different ways
# original: the original connectome matrix
# tanh_1: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 1 
# tanh_2: ... scaling factor of 2
# tanh_10: ... scaling factor of 10
# sign: the connectome matrix binarized by the sign function
# shuffled_i: the connectome matrix with the synapse counts rows and column indices shuffled
# measurement_error_i: the connectome matrix with the synapse counts randomly perturbed by a poisson process and sign randomly flipped according to the NT confidence

df_sgn = pd.read_csv('../data/connectome_sgn_cnt.csv', index_col=0)
post_root_id_ind = df_sgn['post_root_id_ind'].values
pre_root_id_ind = df_sgn['pre_root_id_ind'].values
syn_count_sgn = df_sgn['syn_count_sgn'].values
n_neurons = np.max([np.max(post_root_id_ind), np.max(pre_root_id_ind)]) + 1
C_orig = csr_matrix((syn_count_sgn, (post_root_id_ind, pre_root_id_ind)), shape=(n_neurons, n_neurons), dtype='float64')

# Define parameters
k_eigs = 1000
n_shuffles = 5
n_measure_error = 5

# Initialize arrays to store results
eig_decomp_labels = ['original', 'tanh_1', 'tanh_2', 'tanh_10', 'sign'] + [f'shuffled_{i}' for i in range(n_shuffles)] + [f'measurement_error_{i}' for i in range(n_measure_error)]
n_labels = len(eig_decomp_labels)
eigenvalues_all = np.zeros((n_labels, k_eigs))
eigenvectors_all = np.zeros((n_labels, n_neurons, k_eigs))

#make the datatype complex for the eigenvectors and eigenvalues
da_eigvec = xr.DataArray(eigenvectors_all.astype('complex128'), 
                         dims=['transform', 'neuron', 'eig'], coords={'transform': eig_decomp_labels, 'neuron': np.arange(n_neurons), 'eig': np.arange(k_eigs)})
da_eigval = xr.DataArray(eigenvalues_all.astype('complex128'),
                            dims=['transform', 'eig'], coords={'transform': eig_decomp_labels, 'eig': np.arange(k_eigs)})

#%%
# Calculate original eigenvalues and eigenvectors
eigenvalues_orig, eig_vec_orig = sp.linalg.eigs(C_orig, k=k_eigs)
eig_vec_orig = eig_vec_orig[:, np.argsort(np.abs(eigenvalues_orig))[::-1]]
eigenvalues_orig = eigenvalues_orig[np.argsort(np.abs(eigenvalues_orig))[::-1]]

# Store original results using eig_decomp_labels
da_eigval.loc['original'] = eigenvalues_orig
da_eigvec.loc['original'] = eig_vec_orig

#%% shuffling of the weights
#now draw random indices and shuffle the matrix
for i in tqdm(range(n_shuffles)):
    post_root_id_ind_rand = np.random.permutation(post_root_id_ind)
    pre_root_id_ind_rand = np.random.permutation(pre_root_id_ind)
    C_rand = csr_matrix((syn_count_sgn, (post_root_id_ind_rand, pre_root_id_ind_rand)), shape=(n_neurons, n_neurons), dtype='float64')
    eigenvalues_rand, eig_vec_rand = sp.linalg.eigs(C_rand, k=k_eigs)
    eig_vec_rand = eig_vec_rand[:,np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eigenvalues_rand = eigenvalues_rand[np.argsort(np.abs(eigenvalues_rand))[::-1]]
    da_eigval.loc[f'shuffled_{i}'] = eigenvalues_rand
    da_eigvec.loc[f'shuffled_{i}'] = eig_vec_rand
#%% non-linear transformations of the weights
max_syn_cnt = np.max(syn_count_sgn)
C_orig_tanh_1 = np.tanh(C_orig/(max_syn_cnt/1))
C_orig_tanh_2 = np.tanh(C_orig/(max_syn_cnt/2))
C_orig_tanh_10 = np.tanh(C_orig/(max_syn_cnt/10))
#now for extremal case binarize the connectome
C_orig_sign = C_orig.sign()
#matrix list 
C_list = [C_orig_tanh_1, C_orig_tanh_2, C_orig_tanh_10, C_orig_sign]
eig_decomp_nonlin_labels = ['tanh_1', 'tanh_2', 'tanh_10', 'sign']


#%%
for i, C in enumerate(C_list):
    eigenvalues, eig_vec = sp.linalg.eigs(C, k=k_eigs,)
    #sort eigenvalues by magnitude
    max = np.max(np.abs(eigenvalues))
    #sort eigenvectors by magnitude
    eig_vec = eig_vec[:,np.argsort(np.abs(eigenvalues))[::-1]]
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    da_eigval.loc[eig_decomp_nonlin_labels[i]] = eigenvalues
    da_eigvec.loc[eig_decomp_nonlin_labels[i]] = eig_vec

#%% now measurement error model
df_sgn = pd.read_csv('../data/connectome_sgn_cnt_prob.csv', index_col=0)
p_exc = df_sgn.groupby('pre_root_id_ind').agg({'p_exc': 'first'})
df_sgn_rand = df_sgn.copy()
for i in tqdm(range(n_measure_error)):
    rand_sgn = np.random.binomial(1, p_exc['p_exc'].values)
    rand_sgn[rand_sgn == 0] = -1
    p_exc['rand_sgn'] = rand_sgn
    df_sgn_rand['sgn'] = df_sgn_rand['pre_root_id_ind'].map(p_exc['rand_sgn'])
    syn_count = np.abs(df_sgn_rand['syn_count_sgn'].values)
    syn_count = np.random.poisson(syn_count)
    syn_count_sgn = syn_count * df_sgn_rand['sgn'].values
    syn_count_sgn[np.abs(syn_count_sgn) < 5] = 0
    C_sgn_rand = csr_matrix((syn_count_sgn, (df_sgn_rand['pre_root_id_ind'], df_sgn_rand['post_root_id_ind'])), shape=(n_neurons, n_neurons), dtype='float64')
    eigenvalues_rand, eig_vec_rand = sp.linalg.eigs(C_sgn_rand, k=k_eigs)
    eig_vec_rand = eig_vec_rand[:,np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eigenvalues_rand = eigenvalues_rand[np.argsort(np.abs(eigenvalues_rand))[::-1]]
    da_eigval.loc[f'measurement_error_{i}'] = eigenvalues_rand
    da_eigvec.loc[f'measurement_error_{i}'] = eig_vec_rand

#%% xarray can't save complex numbers to netcdf, so we need to save the real and imaginary parts separately
def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
    return ds.to_netcdf(*args, **kwargs)

def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds['real'] + ds['imag'] * 1j

save_complex(da_eigvec, '../data/eigenvectors_robust.nc')
save_complex(da_eigval, '../data/eigenvalues_robust.nc')

# %%
