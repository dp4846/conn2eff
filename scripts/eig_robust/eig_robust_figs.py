#%% robustness of eigencircuit results
import numpy as np
import matplotlib.pyplot as plt
#import sparse matrix package
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import xarray as xr
import matplotlib.gridspec as gridspec
#%%
#
def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
    return ds.to_netcdf(*args, **kwargs)

def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds['real'] + ds['imag'] * 1j


#load the above
da_eigvec = xr.open_dataset('../../data/eigenvectors_robust.nc')
da_eigval = read_complex('../../data/eigenvalues_robust.nc')

#%%
i = 0
eigenvalues_orig = da_eigval.loc['original'].values
#eig_vec_orig = da_eigvec.loc['original'].values
eigenvalues_rands = [da_eigval.loc[f'shuffled_{i}'].values for i in range(5)]
#eig_vec_rands = da_eigvec.loc[f'shuffled_{i}'].values
k_eigs = eigenvalues_orig.shape[0]
plt.figure()
ind = np.arange(1, k_eigs+1)
plt.plot(ind, np.abs(eigenvalues_orig), c='k', label='Original',)
for i in range(5):
    plt.plot(ind, np.abs(eigenvalues_rands[i]), c='gray', label='Shuffle', ls='--')
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.legend(['Original', 'Shuffled'])
plt.loglog()
#make square plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.ylim(1e1, 1e4)
plt.title('Eigenvalues of connectome')
plt.savefig('./eigenvalues_orig_shuffled.png', dpi=300)
#%%
#make five subplots of the eigenvectors for the original and shuffled connectomes
eig_indices = [0, 4, 9, 49, 99]  # Indices for 1st, 5th, 10th, 50th, 100th eigenvectors
plt.figure(figsize=(10, 12))
for i, eig_index in enumerate(eig_indices):
    plt.subplot(5, 2, 2*i+1)
    plt.plot(da_eigvec['real'].loc['original'][:, eig_index], label='Real')
    plt.plot(da_eigvec['imag'].loc['original'][:, eig_index], label='Imaginary')
    
    if i != 4:
       plt.gca().set_xticklabels([])
       plt.gca().set_yticklabels([])
    else:
        plt.xlabel('Neuron index')
        plt.ylabel('Eigenvector value')
    plt.title('Original eigenvector ' + str(eig_index+1))
    if i == 0:
        plt.legend(loc='best')
    ylim1 = plt.ylim()

    plt.subplot(5, 2, 2*i+2)
    plt.plot(da_eigvec['real'].loc['shuffled_1'][:, eig_index], label='Real')
    plt.plot(da_eigvec['imag'].loc['shuffled_1'][:, eig_index], label='Imaginary')
    plt.title('Shuffled connectome eigenvector ' + str(eig_index+1))
    ylim2 = plt.ylim()
    if i != 4:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
    # Set y limits to be symmetric and equal between the two plots
    max_abs = np.max([np.abs(ylim1[0]), np.abs(ylim1[1]), np.abs(ylim2[0]), np.abs(ylim2[1])])
    ylim = (-max_abs, max_abs)
    plt.subplot(5, 2, 2*i+1)
    plt.ylim(-0.5, 0.5)
    plt.subplot(5, 2, 2*i+2)
    plt.ylim(-0.5, 0.5)
    #only if the lower left plot put on xticks
plt.savefig('./eigenvectors_orig_shuffled.png', dpi=300)
#%%
#need a simple function to calculate sorted eigenvectors power up to certain cumulative quantile
def eigvec_power(eig_vec, quantile):
    eig_power = np.abs(eig_vec)**2
    eig_power_sort = np.sort(eig_power)[::-1]
    eig_power_cumsum = np.cumsum(eig_power_sort)
    ind = np.where(eig_power_cumsum > quantile*eig_power_cumsum[-1])[0][0]
    return eig_power_sort, eig_power_cumsum, ind

#get the ind for the 75th quantile for the original and shuffled connectomes and plot them
quantile = 0.75
inds = []
for eig_index in tqdm(range(k_eigs)):
    eig_power_sort_orig, eig_power_cumsum_orig, ind_orig = eigvec_power(da_eigvec['real'].loc['original'][:, eig_index], quantile)
    eig_power_sort_rand, eig_power_cumsum_rand, ind_rand = eigvec_power(da_eigvec['real'].loc['shuffled_1'][:, eig_index], quantile)
    inds.append([ind_orig, ind_rand])
inds = np.array(inds)
#%%
ind = np.arange(1, k_eigs+1)
plt.scatter(ind, inds[:,0], label='Original', c='k')
plt.scatter(ind, inds[:,1], label='Shuffled', c='gray')
plt.xlabel('Eigenvalue index')
plt.ylabel('Number of loadings to reach ' + str(quantile) + ' quantile power')
plt.legend(loc='lower right')
plt.loglog()
plt.title('Eigenvector concentration')
plt.savefig('./eigenvector_sparsity_orig_shuffled.png', dpi=300)
#%%
#make two subplots of the real imaginary axis for eigenvalues for the original and shuffled connectomes
plt.figure()
ylim = (-1.1, 1.1)
ticks = [-1, 0, 1]

eigenvalues_orig = da_eigval.loc['original'].values/np.max(np.abs(da_eigval.loc['original'].values))
eigenvalues_rand = da_eigval.loc['shuffled_1'].values/np.max(np.abs(da_eigval.loc['original'].values))
# Original eigenvalues plot
plt.subplot(1, 3, 1)
plt.scatter(np.real(eigenvalues_orig), np.imag(eigenvalues_orig), c='k', s=5)
plt.title('Original eigenvalues')
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.ylim(ylim); plt.xlim(ylim); plt.yticks(ticks); plt.xticks(ticks)
plt.xlabel('Real'); plt.ylabel('Imaginary')

# Shuffled eigenvalues plot
plt.subplot(1, 3, 2)
plt.scatter(np.real(eigenvalues_rand), np.imag(eigenvalues_rand), c='k', s=5)
plt.title('Shuffled eigenvalues')
plt.gca().set_aspect('equal', adjustable='box')

plt.ylim(ylim); plt.xlim(ylim); plt.yticks(ticks); plt.xticks(ticks)
plt.gca().set_yticklabels([]); plt.gca().set_xticklabels([]);
plt.grid()

# Shuffled eigenvalues scaled plot
eigenvalues_rand_scaled = eigenvalues_rand/np.max(np.abs(eigenvalues_rand))
plt.subplot(1, 3, 3)
plt.scatter(np.real(eigenvalues_rand_scaled), np.imag(eigenvalues_rand_scaled), c='k', s=5)
plt.title('Shuffled eigenvalues\nscaled')

plt.gca().set_aspect('equal', adjustable='box')
plt.ylim(ylim); plt.xlim(ylim); plt.yticks(ticks); plt.xticks(ticks)
plt.gca().set_yticklabels([]); plt.gca().set_xticklabels([]);
plt.grid()
plt.tight_layout()
plt.savefig('./eigenvalues_orig_shuffled.png', dpi=300)
#%%
def process_eigenvectors(eig_vec):
    eig_vecs_real = []
    j=0
    while j < len(eig_vec[0]):
        ev = eig_vec[:, j]
        j+=1
        if np.sum(np.imag(ev)**2)>0:
            j+=1
            eig_vecs_real.append(np.imag(ev))
            eig_vecs_real.append(np.real(ev))
        else:
            eig_vecs_real.append(np.real(ev))

    eig_vecs_real = np.array(eig_vecs_real).T
    return eig_vecs_real


#%%
#get new eigenvectors for original and shuffled connectomes
# eig_vec_orig_no_conj = process_eigenvectors(eig_vec_orig)
# eig_vec_rand_no_conj = process_eigenvectors(eig_vec_rand)
eig_vec_orig = da_eigvec['real'].loc['original'].values + 1j*da_eigvec['imag'].loc['original'].values
eig_vec_rand = da_eigvec['real'].loc['shuffled_1'].values + 1j*da_eigvec['imag'].loc['shuffled_1'].values
eig_vec_orig_no_conj = process_eigenvectors(eig_vec_orig)
eig_vec_rand_no_conj = process_eigenvectors(eig_vec_rand)
#%%
#truncate to min len of eigenvectors


#plot correlation matrix of eigenvectors for original and shuffled connectomes
#correlation matrix for original connectome
corr_orig = np.abs(np.corrcoef(eig_vec_orig_no_conj.T))
#correlation matrix for shuffled connectome
corr_rand = np.abs(np.corrcoef(eig_vec_rand_no_conj.T))

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # 2 plots and 1 colorbar

ax0 = plt.subplot(gs[0])
im0 = ax0.imshow(corr_orig[:50,:50], cmap='gray')
ax0.set_title('Original connectome')

ax1 = plt.subplot(gs[1])
im1 = ax1.imshow(corr_rand[:50,:50], cmap='gray')
ax1.set_title('Shuffled connectome')

ax2 = plt.subplot(gs[2])
fig.colorbar(im1, cax=ax2)  # Add colorbar, use the color scale of the second plot
#set title of colorbar to abs value of correlation
ax2.set_ylabel('|r|', rotation=0, labelpad=10, fontsize=12)

#%%
s=2
fig, axs = plt.subplots(1, 2, figsize=(4*s,2*s), dpi=300)
for i, eig_cov_copy in enumerate([corr_orig, corr_rand]):
    ax = axs[i]
    eig_cov_copy[np.diag_indices_from(eig_cov_copy)] = 0
    n = len(eig_cov_copy)

    # plot quantile 0.5, 0.9 and 0.99
    ax.scatter(range(1, n+1),np.quantile(np.abs(eig_cov_copy),0.5, 1), 
                            c='k',  label='median',s=1, rasterized=True)
    ax.scatter(range(1, n+1),np.quantile(np.abs(eig_cov_copy),0.80, 1),
        c='g', label='80th percentile',s=1, rasterized=True)
    ax.scatter(range(1, n+1),np.quantile(np.abs(eig_cov_copy),0.99, 1),
        c='r', label='99th percentile',s=1, rasterized=True)
    ax.scatter(range(1,n+1),np.max(np.abs(eig_cov_copy), 1), c='b', 
                    label='max',s=1, rasterized=True)
    ax.set_ylim(-0.01,1)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

#make it so that only the left plot has y and x labels and ticklabels and a legend
axs[0].set_ylabel('Eigenvector correlation |r|')
axs[0].set_xlabel('Eigenvalue rank')
axs[0].legend()

axs[1].set_yticklabels([])
axs[1].set_xticklabels([])
#set title
axs[0].set_title('Original connectome')
axs[1].set_title('Shuffled connectome')


#%%
plt.figure()
ind = np.arange(1, k_eigs+1)
#make a reasonable color scheme colors are different transfomr types
colors = ['k', 'gray', 'r', 'b', 'g', 'c']
for i, eigenvalues in enumerate(da_eigval):
    label = str(eigenvalues.coords['transform'].values )
    print(label)
    ls = '-'
    if 'shuffled' in label:
        ls = ':'
    if 'measurement_error' in label:
        ls = '--'
    if 'shuffled' in label or 'measurement_error' in label:
        c = 'gray'
    elif 'original' in label:
        c = 'k'
    elif 'tanh_2' in label:
        c = 'orange'
    elif 'tanh_10' in label:
        c = 'red'
    elif 'tanh_1' in label:
        c = 'pink'
    elif 'sign' in label:
        c = 'c'
    else:
        c = 'gray'
    print(ls)
    plt.plot(ind, np.abs(eigenvalues)/np.abs(eigenvalues[0]), label=label, c=c, ls=ls)

plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.legend(loc=(1.05, 0))
plt.loglog()
#make square plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.ylim(1e-2, 1.1)
plt.title('Eigenvalues of connectome')
plt.tight_layout()
plt.savefig('eigenvalues_orig_shuffled_tanh_sign.png', dpi=300)
# %%
def process_eigenvectors_real_imag_split(eig_vec):
    eig_vecs_real = []
    eig_vecs_real_ind = []
    eig_vecs_imag = []
    eig_vecs_imag_ind = []
    j=0
    while j < len(eig_vec[0]):
        ev = eig_vec[:, j]

        if np.sum(np.imag(ev)**2)>0:
            eig_vecs_imag_ind.append(j)
            eig_vecs_imag.append([np.imag(ev),np.real(ev)])
            j+=1
        else:
            eig_vecs_real_ind.append(j)
            eig_vecs_real.append(np.real(ev))
        j+=1

    eig_vecs_real = np.array(eig_vecs_real).T
    eig_vecs_imag = np.array(eig_vecs_imag).T
    eig_vecs_real_ind = np.array(eig_vecs_real_ind)
    eig_vecs_imag_ind = np.array(eig_vecs_imag_ind)
    return eig_vecs_real, eig_vecs_imag, eig_vecs_real_ind, eig_vecs_imag_ind

eig_vecs_real, eig_vecs_imag, eig_vecs_real_ind, eig_vecs_imag_ind = process_eigenvectors_real_imag_split(eig_vec_list[0][..., :500])
n_real_orig = len(eig_vecs_real_ind)
R_max = []
for i in range(len(labels)):
    eig_vecs_real_trans, eig_vecs_imag_trans, eig_vecs_real_ind_trans, eig_vecs_imag_ind_trans = process_eigenvectors_real_imag_split(eig_vec_list[i][..., :500])

    
    n_real_trans = len(eig_vecs_real_ind_trans)
    R_real = np.abs(np.corrcoef(eig_vecs_real.T, eig_vecs_real_trans.T))[:n_real_orig, n_real_orig:]
    print(eig_vecs_imag_trans.shape)
    print('')

    n_imag_orig = len(eig_vecs_imag_ind)
    n_imag_orig_trans = len(eig_vecs_imag_ind_trans)
    R_imag = np.zeros((n_imag_orig, n_imag_orig_trans))
    for i in tqdm(range(n_imag_orig)):
        X = eig_vecs_imag[..., i]
        for j in range(n_imag_orig_trans):
            Y = eig_vecs_imag_trans[..., j]
            #regress X on Y
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            #now get r-value
            R = np.corrcoef(Y.ravel(), np.dot(X, beta).ravel())
            R_imag[i, j] = R[0,1]
    R_max.append([R_real.max(1), R_imag.max(1)])


# %%
colors = ['k', 'gray', 'r', 'b', 'g', 'c']
for i, a_R_max in enumerate(R_max):
    plt.subplot(2,1,1)
    plt.plot(a_R_max[0], label=labels[i], c=colors[i])
    plt.title('Real eigenvectors')
    plt.legend(loc=(1,0))
    plt.subplot(2,1,2)
    plt.title('Imaginary eigenvectors')
    plt.plot(a_R_max[1], c=colors[i])
    plt.ylim(0,1.1)

plt.suptitle('Max correlation between original and transformed top 100 connectome eigenvectors')
plt.xlabel('Eigenvalue index (original)')
plt.ylabel('Max correlation |r|')
plt.tight_layout()
plt.savefig('./figs/max_correlation_eigenv.png', dpi=300)

#%%
plt.figure(figsize=(4, 2))
df_sgn = pd.read_csv('../../data/connectome_sgn_cnt_prob.csv', index_col=0)
syn_count_sgn = df_sgn['syn_count_sgn'].values
max_syn_cnt = np.max(np.abs(syn_count_sgn))
syn_count_sgn = syn_count_sgn/(max_syn_cnt/2)
tanh_1 = np.tanh(syn_count_sgn)
plt.scatter(syn_count_sgn, tanh_1, s=1, alpha=0.5)
#plt.hist(syn_count_sgn, bins=1000, cumulative=True, density=True, histtype='step', color='k', label='Cumulative distribution')
#plt.legend()
plt.xlabel('c*Synapse count/max')
plt.ylabel('Tanh(c*Synapse count/max)')
plt.title('Tanh transformation of synapse count c=2')
plt.xlim(-2, 2)
# %%
# look at top eigencircutis
eig_index = 0
transform = 'tanh_2'
eig_vec = da_eigvec['real'].loc[transform][:, eig_index] + da_eigvec['imag'].loc[transform][:, eig_index]*1j
eig_val = da_eigval.loc['original'][eig_index]
fn = '../../data/meta_data.csv'
df_class = pd.read_csv(fn, index_col=0)
#get top abs values of eigenvector
#get number neurons to 75th quantile
quantile = 0.75
eig_power = np.abs(eig_vec)**2
eig_power_sort = np.sort(eig_power)[::-1]
eig_power_cumsum = np.cumsum(eig_power_sort)
top_n = np.where(eig_power_cumsum > quantile*eig_power_cumsum[-1])[0][0]
top_inds = np.argsort(np.abs(eig_vec))[-top_n:].values
top_vals = eig_vec[top_inds]
top_neurons = df_class.iloc[top_inds]
print(top_neurons)
# %%
