#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import xarray as xr
from tqdm import tqdm
# xarray can't save complex numbers to netcdf, so we need to save the real and imaginary parts separately
def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
    return ds.to_netcdf(*args, **kwargs)
def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds['real'] + ds['imag'] * 1j
np.random.seed(42)
#%%
# this script generates several different eigendecompositions of the connectome matrix transformed in different ways
# original: the original connectome matrix
# tanh_1: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 1 
# tanh_2: ... scaling factor of 2
# tanh_10: ... scaling factor of 10
# sign: the connectome matrix binarized by the sign function
# shuffled_i: the connectome matrix with the synapse counts rows and column indices shuffled
# measurement_error_i: the connectome matrix with the synapse counts randomly perturbed by a poisson process and sign randomly flipped according to the NT confidence

df_sgn = pd.read_csv('../../../data/connectome_sgn_cnt_prob.csv', index_col=0)
post_root_id_ind = df_sgn['post_root_id_ind'].values
pre_root_id_ind = df_sgn['pre_root_id_ind'].values
syn_count_sgn = df_sgn['syn_count_sgn'].values
n_neurons = np.max([np.max(post_root_id_ind), np.max(pre_root_id_ind)]) + 1
C_orig = csr_matrix((syn_count_sgn, (post_root_id_ind, pre_root_id_ind)), shape=(n_neurons, n_neurons), dtype='float64')

k_eigs = 1000 #number of eigenvalues/eigenvectors to calculate
n_shuffles = 5 #number of shuffles to perform
n_measure_error = 5 #number of measurement error models to generate

# labels for all transformations of the connectome matrix
eig_decomp_labels = (['original', 'tanh_1', 'tanh_2', 'tanh_10', 'sign'] + 
                    [f'shuffled_{i}' for i in range(n_shuffles)] + 
                    [f'measurement_error_{i}' for i in range(n_measure_error)] +
                    [f'measurement_error_norm_sd={scale}_sim={i}' for i in range(n_measure_error) for scale in [0.01, 0.1, 1,]])

#%% initialize arrays to store eigenvalues and eigenvectors
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
#store the original eigenvalues and eigenvectors
da_eigval.loc['original'] = eigenvalues_orig
da_eigvec.loc['original'] = eig_vec_orig

#%% shuffling transformations
# draw random indices and shuffle the matrix
for i in tqdm(range(n_shuffles)):
    post_root_id_ind_rand = np.random.permutation(post_root_id_ind)
    pre_root_id_ind_rand = np.random.permutation(pre_root_id_ind)
    C_rand = csr_matrix((syn_count_sgn, (post_root_id_ind_rand, pre_root_id_ind_rand)), shape=(n_neurons, n_neurons), dtype='float64')
    eigenvalues_rand, eig_vec_rand = sp.linalg.eigs(C_rand, k=k_eigs)
    eig_vec_rand = eig_vec_rand[:,np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eigenvalues_rand = eigenvalues_rand[np.argsort(np.abs(eigenvalues_rand))[::-1]]
    da_eigval.loc[f'shuffled_{i}'] = eigenvalues_rand
    da_eigvec.loc[f'shuffled_{i}'] = eig_vec_rand
#%% non-linear transformations 
max_syn_cnt = np.max(syn_count_sgn)
C_orig_tanh_1 = np.tanh(C_orig/(max_syn_cnt/1))
C_orig_tanh_2 = np.tanh(C_orig/(max_syn_cnt/2))
C_orig_tanh_10 = np.tanh(C_orig/(max_syn_cnt/10))
#now for extremal case binarize the connectome
C_orig_sign = C_orig.sign()
#matrix list 
C_list = [C_orig_tanh_1, C_orig_tanh_2, C_orig_tanh_10, C_orig_sign]
eig_decomp_nonlin_labels = ['tanh_1', 'tanh_2', 'tanh_10', 'sign']

#now perform eigendecomposition for each of these matrices
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

#%% 

save_complex(da_eigvec, '../../../data/eigenvectors_robust.nc')
save_complex(da_eigval, '../../../data/eigenvalues_robust.nc')


# %%generates data and figure for ED fig 9a where we vary SNR of the connectome measurement with a gaussian
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

k_eigs = 500
df_sgn = pd.read_csv('../../../data/connectome_sgn_cnt_prob.csv', index_col=0)
df_sgn_rand = df_sgn.copy()
C_sgn_orig = csr_matrix((df_sgn['syn_count_sgn'].values, (df_sgn['post_root_id_ind'], df_sgn['pre_root_id_ind'])), shape=(n_neurons, n_neurons), dtype='float64')
eigenvalues_orig, eig_vec_orig = sp.linalg.eigs(C_sgn_orig, k=k_eigs)
eig_vec_orig = eig_vec_orig[:, np.argsort(np.abs(eigenvalues_orig))[::-1]]
eigenvalues_orig = eigenvalues_orig[np.argsort(np.abs(eigenvalues_orig))[::-1]]
eig_vecs_real, eig_vecs_imag, eig_vecs_real_ind, eig_vecs_imag_ind = process_eigenvectors_real_imag_split(eig_vec_orig[:, :k_eigs])
n_real_orig = len(eig_vecs_real_ind)
R_max = []
for var_scale in [0.01, 0.1, 1]:
    syn_count = df_sgn['syn_count_sgn'].values
    syn_count = np.random.normal(syn_count, np.abs(syn_count*var_scale)**0.5)
    syn_count[np.abs(syn_count) < 5] = 0
    C_sgn_rand = csr_matrix((syn_count, (df_sgn['post_root_id_ind'], df_sgn['pre_root_id_ind'])), shape=(n_neurons, n_neurons), dtype='float64')
    eigenvalues_rand, eig_vec_rand = sp.linalg.eigs(C_sgn_rand, k=k_eigs)
    eig_vec_rand = eig_vec_rand[:,np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eigenvalues_rand = eigenvalues_rand[np.argsort(np.abs(eigenvalues_rand))[::-1]]
    eig_vecs_real_trans, eig_vecs_imag_trans, eig_vecs_real_ind_trans, eig_vecs_imag_ind_trans = process_eigenvectors_real_imag_split(eig_vec_rand[:, :n_eigs])
    n_real_trans = len(eig_vecs_real_ind_trans)
    #get correlation matrix between original and transformed eigenvectors that are real
    R_real = np.abs(np.corrcoef(eig_vecs_real.T, eig_vecs_real_trans.T))[:n_real_orig, n_real_orig:]
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
#%%
colors = ['k', 'gray', 'r', 'b', 'g', 'c']
nice_labels = ['100', '10', '1']
plt.figure(figsize=(4,3), dpi=300)
for i, a_R_max in enumerate(R_max):
    plt.subplot(2,1,1)
    eig_inds = np.arange(1, len(a_R_max[0])+1)
    plt.plot(eig_inds, a_R_max[0], label=nice_labels[i], c=colors[i])
    plt.title('Real eigenvectors')
    plt.legend(loc=(1.05,0), title='Connectome\nmeasurement\nSNR')
    plt.ylim(0,1.1)#
    plt.subplot(2,1,2)
    plt.title('Imaginary eigenvectors')
    eig_inds = np.arange(1, len(a_R_max[1])+1)
    plt.plot(eig_inds, a_R_max[1], c=colors[i])
    plt.ylim(0,1.1)

#plt.suptitle('Max correlation between original and transformed top 100 connectome eigenvectors')
plt.xlabel('Eigenvalue index (original)')
plt.ylabel('Max correlation |r|')
plt.tight_layout()
plt.savefig('./ED_fig_9a_max_correlation_eigenv.png', dpi=300)

# %% eig_corruption will be used to generate the data for ED fig 9b in concert with eig_further_char.py 
corruption = list(R_max[-1][0]) + list(R_max[-1][1])
corruption_ind = list(eig_vecs_real_ind) + list(eig_vecs_imag_ind)
#save this to csv
df_corruption = pd.DataFrame({'corruption':corruption, 'eig_ind':corruption_ind})
df_corruption.to_csv('../../../data/eig_corruption.csv')
