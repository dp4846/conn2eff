#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import xarray as xr
from tqdm import tqdm
np.random.seed(42)
def process_eigenvectors_real_imag_split(eig_vec):
    eig_vecs_real = []  # List to store real parts of eigenvectors
    eig_vecs_real_ind = []  # List to store indices of eigenvectors with real parts
    eig_vecs_imag = []  # List to store imaginary parts of eigenvectors
    eig_vecs_imag_ind = []  # List to store indices of eigenvectors with imaginary parts
    j = 0
    while j < len(eig_vec[0]):
        ev = eig_vec[:, j]
        if np.sum(np.imag(ev)**2) > 0:  # Check if eigenvector has non-zero imaginary part
            eig_vecs_imag_ind.append(j)
            eig_vecs_imag.append([np.imag(ev), np.real(ev)])  # Store imaginary and real parts together
            j += 1
        else:
            eig_vecs_real_ind.append(j)
            eig_vecs_real.append(np.real(ev))  # Store real part only
        j += 1

    eig_vecs_real = np.array(eig_vecs_real).T  # Convert to numpy array and transpose
    eig_vecs_imag = np.array(eig_vecs_imag).T  # Convert to numpy array and transpose
    eig_vecs_real_ind = np.array(eig_vecs_real_ind)  # Convert to numpy array
    eig_vecs_imag_ind = np.array(eig_vecs_imag_ind)  # Convert to numpy array
    return eig_vecs_real, eig_vecs_imag, eig_vecs_real_ind, eig_vecs_imag_ind

df_sgn = pd.read_csv('../../data/connectome_sgn_cnt_prob.csv', index_col=0)
post_root_id_ind = df_sgn['post_root_id_ind'].values
pre_root_id_ind = df_sgn['pre_root_id_ind'].values
syn_count_sgn = df_sgn['syn_count_sgn'].values
n_neurons = np.max([np.max(post_root_id_ind), np.max(pre_root_id_ind)]) + 1
C_orig = csr_matrix((syn_count_sgn, (post_root_id_ind, pre_root_id_ind)), shape=(n_neurons, n_neurons), dtype='float64')

k_eigs = 1000
eigenvalues_orig, eig_vec_orig = sp.linalg.eigs(C_orig, k=k_eigs)
eig_vec_orig = eig_vec_orig[:, np.argsort(np.abs(eigenvalues_orig))[::-1]]
eigenvalues_orig = eigenvalues_orig[np.argsort(np.abs(eigenvalues_orig))[::-1]]

eig_vecs_real, eig_vecs_imag, eig_vecs_real_ind, eig_vecs_imag_ind = process_eigenvectors_real_imag_split(eig_vec_orig)
n_real_orig = len(eig_vecs_real_ind)
# %% 
eig_inds = np.array(list(eig_vecs_real_ind) + list(eig_vecs_imag_ind))
#%%
#sort these inds from 0 to k_eigs
eig_inds = np.sort(eig_inds)
#%%
#save eig_inds to csv
df_eig_inds = pd.DataFrame({'eig_inds':eig_inds})
df_eig_inds.to_csv('../../data/unique_eig_inds.csv')

#%%
eig_vals = eigenvalues_orig[eig_inds]
eig_vecs = eig_vec_orig[:, eig_inds]

#get angle in complex plane of each eigenvalue in degrees, map them 0 to 180
eig_angles = np.angle(eig_vals, deg=True)
eig_angles = np.abs(eig_angles)
ind = np.arange(1, len(eig_angles)+1)
plt.figure(figsize=(4, 2), dpi=300)
plt.scatter(ind, eig_angles, s=3, c='k')
plt.semilogx()
plt.ylim(-10,190)
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue angle \n(degrees)')
plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
plt.yticks([0, 45, 90, 135, 180], ['0 (slow/+real)', '45', '90', '135', '180 (fast/-real)'])
#%%
#get ind of highest eignvalue
max_ind = np.argsort((eig_vals))[::-1][0]
#a_ind = 53 had negative eignvalue but top two loadings had positive weights
max_abs_ind = np.argsort(np.abs(eig_vecs[:,max_ind]))[::-1]
top_ind = max_abs_ind[:10]
W = C_orig[top_ind, :][:, top_ind].todense()
vm = np.max(np.abs(W))
plt.figure(figsize=(3, 3), dpi=300)
plt.imshow(W, cmap='coolwarm', vmin=-vm, vmax=vm)
#print the first eigvalue of W
print(eig_vals[53])
print(np.sort(np.linalg.eigvalsh(W))[np.array([-1,0])])
# %%
#mak a histogram of the angles
plt.figure(figsize=(2, 2), dpi=300)
plt.hist(eig_angles, bins=20, color='k')
plt.xlabel('Eigenvalue angle \n(degrees)')
plt.ylabel('Count')
plt.xticks([0, 45, 90, 135, 180], ['0', '45', '90', '135', '180'])

#%%
#get eigenvectors assosciated with eig_inds
eig_circ_syn_counts = []
top_inds = []
for ev_ind in tqdm(range(eig_vecs.shape[1])):
    all_sorted_inds = np.argsort(np.abs(eig_vecs[:, ev_ind]))[::-1]
    #get up to 75% power 
    ev_abs = np.abs(eig_vecs[all_sorted_inds, ev_ind])
    frac_var_ind = np.where((np.cumsum(ev_abs**2)/np.sum(ev_abs**2))>0.75)[0][0]
    if frac_var_ind < 2:
        frac_var_ind = 2
    top_ind = all_sorted_inds[:frac_var_ind]
    #get submatrix of weights from top_inds
    W = C_orig[top_ind, :][:, top_ind].todense()
    #get count of excitatory synapses and inhibitory synapses
    W_ravel = np.ravel(W)
    exc = np.mean(W_ravel[W_ravel>0])
    inh = np.mean(W_ravel[W_ravel<0])
    tot = np.mean(np.abs(W_ravel[np.abs(W_ravel)>0]))
    eig_circ_syn_counts.append([exc, inh, tot])
    top_inds.append(top_ind)
# %%

eig_circ_syn_counts = np.abs(np.array(eig_circ_syn_counts))
plt.figure(figsize=(4, 2), dpi=300)
ind = np.arange(1, len(eig_circ_syn_counts)+1)
plt.plot(ind, eig_circ_syn_counts[:,0],  c='r', label='Excitatory')
plt.plot(ind, eig_circ_syn_counts[:,1],  c='b', label='Inhibitory')
plt.plot(ind, eig_circ_syn_counts[:,2],  c='k', label='Both')
plt.loglog()
#plt.ylim(1e1, 1e3)
plt.xlabel('Eigenvalue index')
plt.ylabel('Synapse sum')
plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
plt.legend(title='Synapse type', loc='lower right', fontsize=4)

#%%
# %%
#plot angles against inhibitory and excitatory synapse counts
#first get fraction of excitatory synapses
frac_exc = []
for top_ind in top_inds[:]:
    W = C_orig[top_ind, :][:, top_ind].todense()
    #get count of excitatory synapses and inhibitory synapses
    W_ravel = np.ravel(W)
    exc = np.sum(W_ravel[W_ravel>0])
    tot = np.sum(np.abs(W_ravel))
    frac_exc.append(exc/tot)

frac_exc = np.array(frac_exc)
s = 0.7
plt.figure(figsize=(3*s, 3*s), dpi=300)
plt.scatter(eig_angles, frac_exc, s=3, c='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('Eigenvalue angle \n(degrees)')
plt.ylabel('Fraction excitatory synapses')
plt.xticks([0, 45, 90, 135, 180], ['0', '45', '90', '135', '180'])
#get ind of highest frac_exc
ind = np.argmax(frac_exc)
top_ind = top_inds[ind]
# plt.figure(figsize=(3, 3), dpi=300)
# W = C_orig[top_ind, :][:, top_ind].todense()
# vm = np.max(np.abs(W))
# plt.imshow(W, cmap='coolwarm', vmin=-vm, vmax=vm)
# plt.colorbar(label='Synapse count')

# %%
#need to get back to root ids
top_inds  = np.array(top_inds)
df_class = pd.read_csv('../../data/meta_data.csv', index_col=0)
df_neurons = pd.read_csv('../../data/' + 'connections.csv')
# Convert pre_root_id and post_root_id into a MultiIndex
df_neurons.set_index(['pre_root_id', 'post_root_id'], inplace=True)
#%%    
root_id_ind = df_class['root_id_ind'].values
root_id = df_class['root_id'].values
n_pilss = []
for a_top_ind in tqdm(top_inds[:]):
    top_root_id = root_id[a_top_ind]
    W = C_orig[a_top_ind, :][:, a_top_ind].todense()
    #for each non-zero synapse, get the pre and post root id
    pre_root_id = []
    post_root_id = []
    # get non-zero synapses coords in W
    non_zero_syn = np.array(np.where(W!=0))
    #get the root ids of the pre and post neurons
    for i in range(non_zero_syn.shape[1]):
        pre_root_id.append(top_root_id[non_zero_syn[1, i]])
        post_root_id.append(top_root_id[non_zero_syn[0, i]])

    pre_root_id = np.array(pre_root_id)
    post_root_id = np.array(post_root_id)
    #use the list zip approach approach to get the neuropil
    index_tuples = list(zip(pre_root_id, post_root_id))
    df_sub = df_neurons.loc[index_tuples]
    n_pils = df_sub['neuropil'].values.tolist()
    n_pilss.append(n_pils)
# %%
loc_indexes = []
for i, a_n_pil in enumerate(n_pilss):
    #get the unique neuropils and counts
    unique, counts = np.unique(a_n_pil, return_counts=True)
    loc_index = np.max(counts)/np.sum(counts)
    loc_indexes.append(loc_index)
#%%
unique, counts = np.unique(n_pilss[0], return_counts=True)
#%%

loc_indexes = np.array(loc_indexes)
plt.figure(figsize=(4, 2), dpi=300)
plt.plot(ind, loc_indexes,  c='k')

plt.semilogx()
plt.xlim(0,1)
plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
plt.xlabel('Eigenvalue index')
plt.ylabel('Fraction synapses\nin one neuropil')
plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
#%% histogram
s= 0.9
plt.figure(figsize=(3*s, 3*s), dpi=300)
plt.hist(loc_indexes, cumulative=True, bins=1000, color='k', histtype='step', density=True)
plt.xlabel('Fraction synapses\nin one neuropil')
plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
plt.ylabel('Fraction of \n eigenvectors')
plt.tight_layout()
# 

#%% ED fig 9b
fn = '../../data/eig_corruption.csv' #from eig_robust_data.py
corruption_ind = pd.read_csv(fn, index_col=0)
print(corruption_ind)
#make another pandas data frame of loc_indexes with the original eig_inds
loc_df = pd.DataFrame({'loc_index':loc_indexes, 'eig_ind':eig_inds})
#on the basis of eig_ind merge the two dataframes
corruption_ind = corruption_ind.merge(loc_df, on='eig_ind')
#plot the correlation between corruption and loc_index
plt.figure(figsize=(3, 3), dpi=300)
plt.scatter(corruption_ind['corruption'], corruption_ind['loc_index'], s=3, c='k')
plt.xlabel('Robustness (r)')
plt.ylabel('Fraction synapses\nin one neuropil')
plt.tight_layout()
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
#calculate spearman correlation
x = corruption_ind['corruption']
y = corruption_ind['loc_index']
from scipy import stats
rho, p = stats.spearmanr(x, y)
print(rho, p)
#add r figure with box
plt.text(0.1, 0.9, f'$\\rho=${rho:.2f}', fontsize=8)
# %%
root_id_ind = df_class['root_id_ind'].values
root_id = df_class['root_id'].values
n_pilss = []
a_top_ind = top_inds[53]
top_root_id = root_id[a_top_ind]
W = C_orig[a_top_ind, :][:, a_top_ind].todense()
#for each non-zero synapse, get the pre and post root id
pre_root_id = []
post_root_id = []
# get non-zero synapses coords in W
non_zero_syn = np.array(np.where(W!=0))
#get the root ids of the pre and post neurons
for i in range(non_zero_syn.shape[1]):
    pre_root_id.append(top_root_id[non_zero_syn[1, i]])
    post_root_id.append(top_root_id[non_zero_syn[0, i]])

pre_root_id = np.array(pre_root_id)
post_root_id = np.array(post_root_id)
#use the list zip approach approach to get the neuropil
index_tuples = list(zip(pre_root_id, post_root_id))
df_sub = df_neurons.loc[index_tuples]
n_pils = df_sub['neuropil'].values.tolist()
# %%
