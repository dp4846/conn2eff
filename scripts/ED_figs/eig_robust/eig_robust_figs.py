#%% robustness of eigencircuit results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr
import matplotlib.gridspec as gridspec

#
def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
    return ds.to_netcdf(*args, **kwargs)

def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds['real'] + ds['imag'] * 1j

#load the above
da_eigvec = read_complex('../../../data/eigenvectors_robust_test.nc')
da_eigval = read_complex('../../../data/eigenvalues_robust_test.nc')
k_eigs = da_eigval.loc['original'].shape[0]


#%% ED fig 10
plt.figure(figsize=(8,8), dpi=300)
ind = np.arange(1, k_eigs+1)
#make a reasonable color scheme colors are different transfomr types
colors = ['k', 'gray', 'r', 'b', 'g', 'c']
for i, eigenvalues in enumerate(da_eigval[:15]):
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
plt.savefig('ED_fig_10_eigenvalues_orig_shuffled_tanh_sign.png', dpi=300)
# %%
# This function splits the given eigenvectors into real and imaginary parts and returns them along with their corresponding indices.
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

k_eigs = 250 #reduced for runtime
labels = ['original', 'tanh_1', 'tanh_2', 'tanh_10', 'sign', 'shuffled_0', ]
eig_vecs_real, eig_vecs_imag, eig_vecs_real_ind, eig_vecs_imag_ind = process_eigenvectors_real_imag_split(da_eigvec.loc['original'].values[:, :k_eigs])
n_real_orig = len(eig_vecs_real_ind)
R_max = []
for i in range(len(labels)):
    _ = (da_eigvec.loc[labels[i]].values)[:,:n_eigs]
    eig_vecs_real_trans, eig_vecs_imag_trans, eig_vecs_real_ind_trans, eig_vecs_imag_ind_trans = process_eigenvectors_real_imag_split(_)
    n_real_trans = len(eig_vecs_real_ind_trans)
    #get correlation matrix between original and transformed eigenvectors that are real
    R_real = np.abs(np.corrcoef(eig_vecs_real.T, eig_vecs_real_trans.T))[:n_real_orig, n_real_orig:]
    print(eig_vecs_imag_trans.shape)

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


# %% ED fig 9d
colors = ['k', 'gray', 'r', 'b', 'g', 'c']
nice_labels = ['Original', 'Tanh c=1', 'Tanh c=2', 'Tanh c=10', 'Sign', 'Shuffled']
for i, a_R_max in enumerate(R_max):
    plt.subplot(2,1,1)
    plt.plot(a_R_max[0], label=nice_labels[i], c=colors[i])
    plt.title('Real eigenvectors')
    plt.legend(loc=(1,0))
    plt.ylim(0,1.1)
    plt.subplot(2,1,2)
    plt.title('Imaginary eigenvectors')
    plt.plot(a_R_max[1], c=colors[i])
    plt.ylim(0,1.1)

plt.suptitle('Max correlation between original and transformed top 100 connectome eigenvectors')
plt.xlabel('Eigenvalue index (original)')
plt.ylabel('Max correlation |r|')
plt.tight_layout()
plt.savefig('ED_fig_9d_max_correlation_eigenv.png', dpi=300)

#%% ED fig 9c
plt.figure(figsize=(4, 2))
df_sgn = pd.read_csv('../../../data/connectome_sgn_cnt_prob.csv', index_col=0)
syn_count_sgn = df_sgn['syn_count_sgn'].values
max_syn_cnt = np.max(np.abs(syn_count_sgn))
syn_count_sgn = syn_count_sgn/(max_syn_cnt/2)
tanh_1 = np.tanh(syn_count_sgn)
plt.scatter(syn_count_sgn, tanh_1, s=1, alpha=0.5)
plt.xlabel('c*Synapse count/max')
plt.ylabel('Tanh(c*Synapse count/max)')
plt.title('Tanh transformation of synapse count c=2')
plt.xlim(-2, 2)
plt.savefig('ED_fig_9c_tanh_transformation.png', dpi=300)
# %%
