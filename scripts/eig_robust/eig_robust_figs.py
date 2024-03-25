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

#first we try shuffling the connectome and seeing if we get similar eigenvalues/eigenvectors
#df_sgn = pd.opetop_dir + 'connectome_sgn_cnt.csv')
#don't load index column
df_sgn = pd.read_csv('../data/connectome_sgn_cnt.csv', index_col=0)
post_root_id_ind = df_sgn['post_root_id_ind'].values
pre_root_id_ind = df_sgn['pre_root_id_ind'].values
syn_count_sgn = df_sgn['syn_count_sgn'].values
n_neurons = np.max([np.max(post_root_id_ind), np.max(pre_root_id_ind)])+1
C_orig = csr_matrix((syn_count_sgn, (post_root_id_ind, pre_root_id_ind)), shape=(n_neurons, n_neurons), dtype='float64')

#get eigenvalues and eigenvectors
k_eigs = 10
eigenvalues_orig, eig_vec_orig = sp.linalg.eigs(C_orig, k=k_eigs,)
eig_vec_orig = eig_vec_orig[:,np.argsort(np.abs(eigenvalues_orig))[::-1]]
eigenvalues_orig = eigenvalues_orig[np.argsort(np.abs(eigenvalues_orig))[::-1]]

n_shuffles = 5
#now draw random indices and shuffle the matrix
eigenvalues_rands = []
eig_vec_rands = []
for i in tqdm(range(n_shuffles)):
    post_root_id_ind_rand = np.random.permutation(post_root_id_ind)
    pre_root_id_ind_rand = np.random.permutation(pre_root_id_ind)
    C_rand = csr_matrix((syn_count_sgn, (post_root_id_ind_rand, pre_root_id_ind_rand)), shape=(n_neurons, n_neurons), dtype='float64')
    eigenvalues_rand, eig_vec_rand = sp.linalg.eigs(C_rand, k=k_eigs)

    eig_vec_rand = eig_vec_rand[:,np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eigenvalues_rand = eigenvalues_rand[np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eigenvalues_rands.append(eigenvalues_rand)
    eig_vec_rands.append(eig_vec_rand)



#%%
plt.figure()
ind = np.arange(1, k_eigs+1)
plt.plot(ind, np.abs(eigenvalues_orig), c='k', label='Original',)
plt.plot(ind, np.abs(eigenvalues_rand), c='gray', label='Shuffle', ls='--')
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.legend()
plt.loglog()
#make square plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.ylim(1e-2, 1.1)
plt.title('Eigenvalues of connectome')
plt.savefig('./figs/eigenvalues_orig_shuffled.png', dpi=300)
#%%
#make five subplots of the eigenvectors for the original and shuffled connectomes
eig_indices = [0, 4, 9, 49, 99]  # Indices for 1st, 5th, 10th, 50th, 100th eigenvectors
plt.figure(figsize=(10, 12))
for i, eig_index in enumerate(eig_indices):
    plt.subplot(5, 2, 2*i+1)
    plt.plot(np.real(eig_vec_orig[:,eig_index]), label='Real')
    plt.plot(np.imag(eig_vec_orig[:,eig_index]), label='Imaginary')
    
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
    plt.plot(np.real(eig_vec_rand[:,eig_index]))
    plt.plot(np.imag(eig_vec_rand[:,eig_index]))
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
plt.savefig('./figs/eigenvectors_orig_shuffled.png', dpi=300)
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
for eig_index in range(k_eigs):
    eig_power_sort_orig, eig_power_cumsum_orig, ind_orig = eigvec_power(eig_vec_orig[:,eig_index], quantile)
    eig_power_sort_rand, eig_power_cumsum_rand, ind_rand = eigvec_power(eig_vec_rand[:,eig_index], quantile)
    inds.append([ind_orig, ind_rand])
inds = np.array(inds)

ind = np.arange(1, k_eigs+1)
plt.scatter(ind, inds[:,0], label='Original', c='k')
plt.scatter(ind, inds[:,1], label='Shuffled', c='gray')
plt.xlabel('Eigenvalue index')
plt.ylabel('Number of loadings to reach ' + str(quantile) + ' quantile power')
plt.legend(loc='upper left')
plt.loglog()
plt.title('Eigenvector sparsity')
plt.savefig('./figs/eigenvector_sparsity_orig_shuffled.png', dpi=300)
#%%
#make two subplots of the real imaginary axis for eigenvalues for the original and shuffled connectomes
plt.figure()
ylim = (-1.1, 1.1)
ticks = [-1, 0, 1]

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
plt.savefig('./figs/eigenvalues_orig_shuffled.png', dpi=300)
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
eig_vec_orig_no_conj = process_eigenvectors(eig_vec_orig)
eig_vec_rand_no_conj = process_eigenvectors(eig_vec_rand)
#truncate to min len of eigenvectors


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


# %%
max_syn_cnt = np.max(syn_count_sgn)
C_orig_tanh_2 = np.tanh(C_orig/(max_syn_cnt/2))
C_orig_tanh_10 = np.tanh(C_orig/(max_syn_cnt/10))
C_orig_tanh_25 = np.tanh(C_orig/(max_syn_cnt/25))
#now for extremal case binarize the connectome
C_orig_sign = C_orig.sign()
#matrix list 
C_list = [C_orig, C_rand, C_orig_tanh_2, C_orig_tanh_10, C_orig_tanh_25, C_orig_sign]

# get eigenvalues and eigenvectors for each matrix
eigenvalues_list = []
eig_vec_list = []
for C in C_list:
    eigenvalues, eig_vec = sp.linalg.eigs(C, k=k_eigs,)
    #sort eigenvalues by magnitude
    max = np.max(np.abs(eigenvalues))
    #sort eigenvectors by magnitude
    eig_vec = eig_vec[:,np.argsort(np.abs(eigenvalues))[::-1]]
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    eigenvalues_list.append(eigenvalues)
    eig_vec_list.append(eig_vec)
#%% sort both eigenvalues and eigenvectors by magnitude
max_syn_cnt = np.max(syn_count_sgn)
plt.subplot(121)
norm = max_syn_cnt/2
plt.scatter(syn_count_sgn/norm, np.tanh(syn_count_sgn/norm), s=1, alpha=0.5)
plt.title('Tanh (2 * C / max(C))')
plt.subplot(122)
norm = max_syn_cnt/25
plt.scatter(syn_count_sgn/norm, np.tanh(syn_count_sgn/norm), s=1, alpha=0.5)
plt.title('Tanh (25 * C / max(C))')
plt.xlim(-10,10)

#%%
plt.figure()
ind = np.arange(1, k_eigs+1)
labels = ['Original', 'Shuffled', 'Tanh 2', 'Tanh 10', 'Tanh 25', 'Sign']
for i, eigenvalues in enumerate(eigenvalues_list):
    plt.plot(ind, np.abs(eigenvalues), label=labels[i])
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.legend()
plt.loglog()
#make square plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.ylim(1e-2, 1.1)
plt.title('Eigenvalues of connectome')
plt.savefig('./figs/eigenvalues_orig_shuffled_tanh_sign.png', dpi=300)
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
# %% now draw noisy samples according to confidence estimates of sign
df_sgn = pd.read_csv('../data/connectome_sgn_cnt_prob.csv', index_col=0)
p_exc = df_sgn.groupby('pre_root_id_ind').agg({'p_exc':'first'})
df_sgn_rand = df_sgn.copy()
n_measure_error = 5
measure_error_eigenvalues = []
measure_error_eigvec = []

for i in tqdm(range(n_measure_error)):
    rand_sgn = np.random.binomial(1, p_exc['p_exc'].values)
    rand_sgn[rand_sgn==0] = -1
    p_exc['rand_sgn'] = rand_sgn
    df_sgn_rand['sgn'] = df_sgn_rand['pre_root_id_ind'].map(p_exc['rand_sgn'])
    syn_count = np.abs(df_sgn_rand['syn_count_sgn'].values)
    syn_count = np.random.poisson(syn_count)
    syn_count_sgn = syn_count*df_sgn_rand['sgn'].values
    #set an abs threshold for 5 synapses
    syn_count_sgn[np.abs(syn_count_sgn)<5] = 0
    C_sgn_rand = csr_matrix((syn_count_sgn, (df_sgn_rand['pre_root_id_ind'], df_sgn_rand['post_root_id_ind'])), shape=(n_neurons, n_neurons), dtype='float64')
    eigenvalues, eig_vec = sp.linalg.eigs(C_sgn_rand, k=k_eigs)
    #sort eigenvalues by magnitude
    eig_vec = eig_vec[:,np.argsort(np.abs(eigenvalues))[::-1]]
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    measure_error_eigenvalues.append(eigenvalues)
    measure_error_eigvec.append(eig_vec)
# %%

ind = np.arange(1, k_eigs+1)
plt.figure()
max = np.max(np.abs(eigenvalues_orig))
plt.plot(ind, np.abs(eigenvalues_orig)/max, label='Original', c='k')
for i in range(n_measure_error):
    plt.plot(ind, np.abs(measure_error_eigenvalues[i])/max, label='Measurement error model', c='gray', ls='--')
plt.loglog()
plt.axis('square')
plt.xlim(1, 1000)
plt.ylim(1e-3, 1.1)
plt.grid()
plt.title('Eigenvalues of connectome')
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.legend(['Original', 'Measurement error model sims'])
plt.savefig('./figs/eigenvalues_orig_measurement_error.png', dpi=300)
# %%
#need to save all eigenvectors in one data array

eig_decomp_labels = ['original','tanh_2', 'tanh_10', 'tanh25', 'sign'] +  ['shuffled_' + str(i) for i in  np.arange(n_shuffles)] + ['measurement_error_' + str(i) for i in  np.arange(n_measure_error)]
n_eig_decomps = len(eig_decomp_labels)
n_neurons = len(eig_vec_list[0])
da = xr.DataArray(np.zeros((n_eig_decomps, n_neurons, k_eigs)), dims=['eig_decomp', 'neuron_index', 'eig_index'],
                  coords={'eig_decomp':eig_decomp_labels, 'neuron_index':np.arange(n_neurons), 'eig_index':np.arange(k_eigs)})

eig_vec_array = np.array(eig_vec_list)
da[0] = eig_vec_array[0]
da[1:5] = eig_vec_array[2:]
da[5:5+n_shuffles] = np.array(eig_vec_rands)
da[5+n_shuffles:] = np.array(measure_error_eigvec)

da.to_netcdf('./data/eigenvectors_robust.nc')

#now eigenvalues
da_eig = xr.DataArray(np.zeros((n_eig_decomps, k_eigs)), dims=['eig_decomp', 'eig_index'],
                  coords={'eig_decomp':eig_decomp_labels, 'eig_index':np.arange(k_eigs)})
da_eig[0] = eigenvalues_orig
da_eig[1:5] = [eigenvalues_list[i] for i in range(2,5)]
da_eig[5:5+n_shuffles] = np.array(eigenvalues_rands)
da_eig[5+n_shuffles:] = np.array(measure_error_eigenvalues)
da_eig.to_netcdf('./data/eigenvalues_robust.nc')
# %%
