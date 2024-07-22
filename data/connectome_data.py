#%% 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp
from tqdm import tqdm

np.random.seed(0)
neurotransmitter_effects = {
    'ACH': 1,    # Excitatory
    'DA': 1,     # Excitatory (can have inhibitory effects in specific brain regions)
    'GABA': -1,  # Inhibitory
    'GLUT': -1,   # Inhibitory (but excitatory in mammals!)
    'OCT': -1,   # Inhibitory
    'SER': -1    # Inhibitory (can have excitatory effects in specific brain circuits)
    }
conn_type = ''
conn_type = '_no_threshold'
top_dir = '../data/'
df_connections = pd.read_csv(top_dir + 'connections' + conn_type + '.csv')
df_class = pd.read_csv(top_dir + 'classification.csv')
#%%
#check if there are any duplicate rows in root_id
#print(df_class.duplicated(subset='root_id').sum())

df_neurons = pd.read_csv(top_dir + 'neurons.csv')
#%%
exc_colums = ['da_avg', 'ach_avg']
df_neurons['exc'] = df_neurons[exc_colums].sum(axis=1)
df_connections['p_exc'] = df_connections['pre_root_id'].map(df_neurons.set_index('root_id')['exc'])
df_connections['nt_sign'] = 0
df_connections['nt_sign'][df_connections['p_exc']>=0.5] = 1
df_connections['nt_sign'][df_connections['p_exc']<0.5] = -1 

df_connections['syn_cnt_sgn'] = df_connections['syn_count']*df_connections['nt_sign']#multiply count by sign for unscaled 'effectome'
df_connections = df_connections.drop(columns='neuropil')#no need for neuropil

#%%
# #%% look at example repeats of pre and post pairs
# g = df_connections.groupby(['pre_root_id', 'post_root_id'])
# duplicate_pairs = g.filter(lambda x: len(x) > 1)
# print(duplicate_pairs.sort_values(by=['pre_root_id', 'post_root_id']))
#%% 
#group by pre_root_id and post_root_id and sum syn_cnt_sgn and syn_count
df_connections = df_connections.groupby(['pre_root_id', 'post_root_id']).agg({'syn_cnt_sgn': 'sum', 'syn_count': 'sum', 'nt_type':'first', 'p_exc':'first'}).reset_index()#sum synapses across all unique pre and post pairs (data is broken by neuropil)
# create new column called pre_root_id_ind and post_root_id_ind for ease of indexing
#unique indices  across pre and post-root ids
unique_root_ids = np.unique(list(df_connections['pre_root_id'].values) + list(df_connections['post_root_id'].values))
# list unique values from 0-n_neurons and put in a dictionary
conv_dict = {val:i for i, val in enumerate(unique_root_ids)}
# go in order through pre and post root ids and and get the simple index from the dictionary
df_connections['pre_root_id_ind'] = df_connections['pre_root_id'].map(conv_dict)
df_connections['post_root_id_ind'] = df_connections['post_root_id'].map(conv_dict)

#%%
n_neurons = len(unique_root_ids)#total rows of full dynamics matrix
syn_count = df_connections['syn_count'].values
is_syn = (syn_count>0).astype('int')
syn_count_sgn = df_connections['syn_cnt_sgn'].values
pre_root_id_ind = df_connections['pre_root_id_ind'].values
post_root_id_ind = df_connections['post_root_id_ind'].values
# unscaled dynamics matrix
C_orig = csr_matrix((syn_count_sgn, (post_root_id_ind, pre_root_id_ind)), shape=(n_neurons, n_neurons), dtype='float64')
#%%
eigenvalues, eig_vec = eigs(C_orig, k=1)#get eigenvectors and values (only need first for scaling)
scale_orig = 0.99/np.abs(eigenvalues[0])#make just below stability
#scale dynamics matrix by largest eigenvalue so that activity decays
W_full = C_orig*scale_orig
#%%
# dictionary to go back to original ids of cells
conv_dict_rev = {v:k for k, v in conv_dict.items()}
root_id = [conv_dict_rev[i] for i in range(n_neurons)]
#save conv_dict_rev as csv 
df_conv = pd.DataFrame.from_dict(conv_dict_rev, orient='index')
df_conv.to_csv(top_dir + 'C_index_to_rootid' + conn_type + '.csv')
#%%%

df_class['root_id'] = df_class['root_id'].astype(int)
df_class['root_id_ind'] = df_class['root_id'].map(conv_dict)
#move root_id_ind to first column
cols = list(df_class.columns)
cols = [cols[-1]] + cols[:-1]
df_class = df_class[cols]
#drop the nan rows, which are the neurons that didn't have any synapses
df_class = df_class.dropna(subset=['root_id_ind'])
df_class['root_id_ind'] = df_class['root_id_ind'].astype(int) 
#sort by root_id_ind
df_class = df_class.sort_values(by='root_id_ind')
df_class = df_class.reset_index(drop=True)
df_class.to_csv(top_dir + 'meta_data' + conn_type + '.csv')
#%%
#make a dataframe from syn_count_sgn, pre_root_id_ind, post_root_id_ind, p_exc
df_sgn = pd.DataFrame({'syn_count_sgn':syn_count_sgn, 'pre_root_id_ind':pre_root_id_ind,   'post_root_id_ind':post_root_id_ind, 'p_exc':df_connections['p_exc'].values})
df_sgn.to_csv(top_dir + 'connectome_sgn_cnt_prob' + conn_type + '.csv')
#save the sparse matrix
sp.sparse.save_npz(top_dir + 'connectome_sgn_cnt_sp_mat' + conn_type + '.npz', C_orig)
sp.sparse.save_npz(top_dir + 'connectome_sgn_cnt_scaled_sp_mat' + conn_type + '.npz', W_full)

#%%
k_eig_vecs = 1000 # number of eigenvectors to use
eigenvalues, eig_vec = eigs(C_orig, k=k_eig_vecs)#get eigenvectors and values (only need first for scaling)
#save the eigenvalues and eigenvectors
np.save(top_dir + 'eigenvalues_' + conn_type + '_' + str(k_eig_vecs) + '.npy', eigenvalues)
np.save(top_dir + 'eigvec_' + conn_type + '_' + str(k_eig_vecs) + '.npy', eig_vec)
# %%
