#%% import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp
from tqdm import tqdm
import matplotlib as mpl
#set the matplotlib font to arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#%% load up dynamics matrix, meta data, eigenvalues, and eigenvectors
top_dir = '../../../'
data_dir = top_dir + 'data/'
C_orig = sp.sparse.load_npz(data_dir + 'connectome_sgn_cnt_sp_mat.npz')
eigenvalues = np.load(data_dir + 'eigenvalues_1000.npy')
eig_vec = np.load(data_dir + 'eigvec_1000.npy')
uniq_eig_inds = np.load(data_dir + 'uniq_eig_inds_1000.npy')
conv_rev = pd.read_csv(data_dir + 'C_index_to_rootid.csv')
conv_dict_rev = dict(zip(conv_rev.iloc[:, 0].values, conv_rev.iloc[:, 1].values,))

#%% Fig 3A,B
fontsize_title = 6
fontsize_label = 6
fontsize_tick = 5
scale = 1/np.abs(eigenvalues[0])
s = 2.5
plt.figure(figsize=(s,0.5*s), dpi=300)
plt.subplot(121)
eig_mag = np.abs(eigenvalues)*scale
sort_ind = np.argsort(eig_mag)[::-1]
eig_mag = eig_mag[sort_ind]
eig_ind = range(1, len(eigenvalues)+1)
plt.plot(eig_ind, eig_mag, label='Effectome prior', c='k', rasterized=False, lw=1)
plt.loglog()
plt.xlim(.5, 3e3)
plt.ylim(.5*scale, 3e3*scale)
plt.xticks([1, 1e1, 1e2, 1e3,], fontsize=fontsize_tick)
plt.yticks([1e-3, 1e-2, 1e-1, 1,], fontsize=fontsize_tick)
plt.gca().set_aspect('equal', adjustable='box')
#plt.grid()
plt.xlabel('Eigenvalue rank', fontsize=fontsize_label)
plt.ylabel(r'$|\lambda_i|$', fontsize=fontsize_label)
log_eig_ind = np.log(eig_ind)
sub_inds = np.logspace(0, 3, 100)
X = np.vstack([np.log(sub_inds), np.ones_like(sub_inds)]).T
b = np.linalg.lstsq(X, np.log(eig_mag[sub_inds.astype(int)-1]), rcond=None)[0]
plt.plot(eig_ind, np.exp(b[0]*log_eig_ind + b[1]), 'red', ls='--', 
                    label='Fit power-law ' +  r'$(\alpha=$' + str(b[0].round(2)) + ')', 
                    alpha=0.7, lw=1)
#plt.legend(fontsize=7, framealpha=1)

# B
plt.subplot(122)
#make inside of each point black and the edge white
plt.scatter(np.real(eigenvalues[sort_ind]*scale)[::-1], np.imag(eigenvalues[sort_ind][::-1])*scale, label='imag', marker='.', s=7, c='k', rasterized=False,
            edgecolors='w', lw=0.1)

lim = np.max(eig_mag)*1.1
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
#put a grid only through origin
plt.axvline(0, c='grey', alpha=0.5, lw=0.5)
plt.axhline(0, c='grey', alpha=0.5, lw=0.5)
yticks = xticks = np.array([-1, -0.5, 0, 0.5, 1])
plt.xticks(xticks, xticks, fontsize=fontsize_tick)
plt.yticks(yticks, yticks, fontsize=fontsize_tick)
#set aspect ratio to 1
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Re' r'$(\lambda_i$)', fontsize=fontsize_label)
plt.ylabel('Im' r'$(\lambda_i$)', fontsize=fontsize_label)
#plt.grid(True, )
plt.tight_layout()
plt.savefig('./eigenvalues_1000.pdf', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.1)

#%% fraction of neurons to q proportion of power
q = 0.75 # fraction of power
inds_cum = []
j=0
js = []
for j in uniq_eig_inds:#just use unique eigenvalues
    eig = eigenvalues[j]
    ev = eig_vec[:, j]
    inds = np.argsort(np.abs(ev))[::-1]
    ev = ev[inds]
    ev = np.abs(ev)**2
    ev = ev/np.sum(ev)
    ev_cum = np.cumsum(ev)
    ind = np.argmax(ev_cum>q)
    inds_cum.append(ind)
#%% FIG 3F
s = 1.
plt.figure(figsize=(s, s))
plt.scatter(range(1, len(inds_cum)+1), np.array(inds_cum), c='k', s=7, edgecolors='w', lw=0.1, rasterized=False)
plt.loglog()
plt.xlabel('Eigenvalue rank', fontsize=fontsize_label)
plt.ylabel('Number of neurons\nto 75% of magnitude', fontsize=fontsize_label)
plt.ylim(1,1e4)
#make the ylabels 1, 10, 100, 1000, etc
yticks = [1, 10, 100, 1000, 10000]
#ytick labels with commas
ytick_labels = [f'{x:,}' for x in yticks]
plt.yticks(yticks, ytick_labels, fontsize=fontsize_tick)
plt.xticks(yticks[:-1], ytick_labels[:-1], fontsize=fontsize_tick)
plt.grid(which='major')
plt.grid(which='minor', alpha=0)
plt.minorticks_off()  # Turn off minor ticks on x-axis
plt.savefig('./eigenvalues_1000_cumsum.pdf', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)

#%% FIG 3E
fontsize_title = 6
fontsize_label = 6
fontsize_tick = 5
s= 1.1
fig, axs = plt.subplots(2,1, figsize=(s,s),)
ev = eig_vec[:, 0]
n_ev = len(ev)
ind = np.arange(n_ev)
lim = 1.2
sub_samp = 1
neuron_index = [0, n_ev//4, n_ev//2, 3*n_ev//4]
for j, i in enumerate([0, 49]):
    max_lim = np.max(np.abs(eig_vec[:,i]))
    axs[j].scatter(ind[::sub_samp], np.real(eig_vec[::sub_samp,i])/max_lim, label='Real', marker='.', s=10, c='k', alpha=1, zorder=100, rasterized=True)
    axs[j].scatter(ind[::sub_samp], np.imag(eig_vec[::sub_samp,i])/max_lim, label='Imag.', marker='.', s=10, c='grey', alpha=1, rasterized=True,)
    axs[j].set_ylim(-lim, lim)
    axs[j].set_xlim(0, n_ev)
    axs[j].set_xticks([1, 60000, 120000])
    axs[j].set_xticklabels(['1', '60,000', '120,000'], fontsize=fontsize_tick)
    axs[j].set_yticklabels(axs[j].get_yticks(), fontsize=fontsize_tick)

    if j == 0:
        #tight legend
        #keep legend off plot
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
    elif j==1:
        #axs[j].legend(loc=(1.01, 0), fontsize=6, labelspacing=0.1, borderpad=0.2, handletextpad=0.1)   
        axs[j].set_xlabel('Neuron index', fontsize=fontsize_label)
        axs[j].set_ylabel('Eigenvector loading', fontsize=fontsize_label)
    else:
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
    #annotate each in upper left above plot with eigenvector index
    #axs[j].annotate('Eigenvector '+str(i+1), xy=(0.01, 1.1), xycoords='axes fraction', fontsize=6, ha='left', va='top')
    #just do it as a title
    axs[j].set_title('Eigenvector '+str(i+1), fontsize=fontsize_label)
plt.tight_layout()
plt.savefig('eigenvector_examples.pdf', transparent=True, dpi=500, bbox_inches='tight', pad_inches=0)
# %%
eig_vecs = []
for j in uniq_eig_inds:
    ev = eig_vec[:, j]
    if np.sum(np.imag(ev)**2) > 0:
        eig_vecs.append(np.imag(ev))
        eig_vecs.append(np.real(ev))
    else:
        eig_vecs.append(np.real(ev))

eig_vecs_real = np.array(eig_vecs)

#%%
eig_vecs_real = eig_vecs_real - np.mean(eig_vecs_real, 1, keepdims=True)
eig_vecs_real = eig_vecs_real/np.sum((eig_vecs_real)**2, 1, keepdims=True)**0.5
eig_cov = eig_vecs_real @ eig_vecs_real.T
#%%FIG 3C
s = 1.1
plt.figure(figsize=(s,s), dpi=300)
plt.imshow(np.abs((eig_cov[:50,:50])), cmap='gray', vmin=0, vmax=1, rasterized=False)
#plt.colorbar( fraction=0.046, pad=0.04, orientation='vertical', ticks=[], aspect=10)
plt.xticks([0,49], fontsize=fontsize_tick)
plt.yticks([0,49], fontsize=fontsize_tick)
ax = plt.gca()
ax.set_xticklabels([1,50], fontsize=fontsize_tick)
ax.set_yticklabels([1,50], fontsize=fontsize_tick)
#plt.xlabel('Eigenvalue rank', fontsize=fontsize_label)
#plt.ylabel('Eigenvalue rank', fontsize=fontsize_label)
plt.tight_layout()
plt.savefig('./eigenvector_correlation.pdf', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)

#%% FIG 3B
s= 1.3
plt.figure(figsize=(s*1.1,s), dpi=300)
eig_cov_copy = eig_cov.copy()
eig_cov_copy[np.diag_indices_from(eig_cov)] = 0
# plt.plot(range(1,1001),np.mean(np.abs(eig_cov_copy),1),c='k', label='mean' )
# 

# plot quantile 0.5, 0.9 and 0.99
ind = range(1,len(eig_vecs_real)+1)
plt.scatter(ind,np.quantile(np.abs(eig_cov_copy),0.5, 1), 
                        c='k',  label='median',s=1, rasterized=False)
plt.scatter(ind,np.quantile(np.abs(eig_cov_copy),0.99, 1),
    c=[0.3,0.3,0.3], label='99th percentile',s=1, rasterized=False)
plt.scatter(ind,np.max(np.abs(eig_cov_copy), 1),c=[0.7,0.7,0.7], 
                label='max',s=1, rasterized=False)
plt.ylim(-0.01,1.1) 
plt.semilogx()
plt.xticks([1, 10, 100, 1000,], fontsize=fontsize_tick)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=fontsize_tick)
plt.gca().set_xticklabels(['1', '10', '100', '1,000', ])
plt.gca().set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
#make background of legend white and markers larger
# plt.legend( loc='upper left', markerscale=3, framealpha=0.7, 
#            labelspacing=0.1, borderpad=0.2, handletextpad=0.1, fontsize=9)
plt.xlabel('Eigenvalue rank', fontsize=fontsize_label)
plt.ylabel('|r|', fontsize=fontsize_label)
#plt.title('Non-normality of dynamics')
plt.tight_layout()
plt.savefig('mode_corr_dist.pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#%%
