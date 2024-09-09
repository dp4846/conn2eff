#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.colors as mcolors
import matplotlib as mpl
from tqdm import tqdm

def rectify(x):
    x[x<0] = 0
    return x

#set the matplotlib font to arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
colors = ['#95A3CE', '#D5A848', '#7dcd13', '#47c4cb', '#47c4cb', '#E760A3', '#BA00FF', '#8B4513', '#39FF14', '#FF4500', '#708090']
#label_colors = [mcolors.to_rgb(color) for color in colors]
#color blind friendly palette up to 11 use viridis
# colormap = plt.cm.cividis
# label_colors = [colormap(a) for a in  np.linspace(0, 1, 10)]
color_palette = {
    "black": "#000000",
    "lightblue": "#B6DBFF",
    "midblue": "#7BB0DF",
    "darkblue": "#1964B0",
    "lightteal": "#00C992",
    "teal": "#008A69",
    "darkteal": "#386350",
    "yellow": "#E9DC6D",
    "orange": "#F4A637",
    "vermilion": "#DB5829",
    "maroon": "#894B45",
    "lightpurple": "#D2BBD7",
    "purple": "#AE75A2",
    "darkpurple": "#882D71",
    "grey": "#DEDEDE"
}
label_colors = [color_palette[key] for key in color_palette.keys()]
label_colors_optic = label_colors[1::3]
label_colors_ring = label_colors[1:]
nms = ["lightblue", "teal", "yellow", "maroon", "darkpurple" ]
label_colors_optic = [color_palette[nm] for nm in nms]
#plot examples of each color in label_colors_optic
s = 1
fig, ax = plt.subplots(1, 1, figsize=(s,s), dpi=300)
for i, color in enumerate(label_colors):
    ax.plot([0,1], [i,i], color=color, label=list(color_palette.keys())[i])
ax.legend(loc='upper right', fontsize=6)
plt.axis('off')
#%% load up dynamics matrix, meta data, eigenvalues, and eigenvectors
top_dir = '../../../'
data_dir = top_dir + 'data/'
C_orig = sp.sparse.load_npz(data_dir + 'connectome_sgn_cnt_sp_mat.npz')
df_meta_W = pd.read_csv(data_dir + 'meta_data.csv')
eigenvalues = np.load(data_dir + 'eigenvalues_1000.npy')
eig_vec = np.load(data_dir + 'eigvec_1000.npy')
scale_orig = 1/np.abs(eigenvalues[0])
nm = data_dir + 'C_index_to_rootid.csv'
conv_rev = pd.read_csv(nm)
conv_dict_rev = dict(zip(conv_rev.iloc[:,0].values, conv_rev.iloc[:,1].values,))
#%%
dt  = 10
sample_rate = 10
T = 200
ts =  np.arange(0, T, 1/sample_rate)
fontsize_title = 6
fontsize_label = 6
fontsize_tick = 5

# Fig 4 optic lobe opponent motion
ev_ind = 0
all_sorted_inds = np.argsort(np.abs(eig_vec[:, ev_ind]))[::-1]
ev_abs = np.abs(eig_vec[all_sorted_inds, ev_ind])
frac_var_ind = np.where(np.cumsum(ev_abs**2)/np.sum(ev_abs**2)>0.75)[0][0]

top_ind = np.array(list(all_sorted_inds[:frac_var_ind]))
#top_ind = [81328, 122580, 111920, 84745] + list(all_sorted_inds[4:frac_var_ind]) #this is wher the elbow is, excluded HSE, and swtiched order for better vis
label = ['Am1', 'LPi21', 'DCH', 'VCH',   ] #easier to hand label
x0 = eig_vec[top_ind, ev_ind]
W = np.array(C_orig[top_ind, :][:, top_ind].todense())
c = 0.63#target amplitude
time_to_c = 0.5#number of steps to reach c
tau = c**(1/time_to_c)
max_abs_eig = np.max(np.abs(np.linalg.eigvals(W)))
scale = (1 - tau)/max_abs_eig
W_scale = scale*W + tau*np.eye(len(top_ind))
W = W/max_abs_eig

A_step = sp.linalg.expm(sp.linalg.logm(W) * 1/(sample_rate*dt))
xs = [x0,]#for recording linear responses
xs_rect = [x0,]#for recording rectified responses
for t in ts[1:]:
    xs.append(A_step @ xs[-1])
    xs_rect.append(A_step @ rectify(xs_rect[-1]))#rectify inputs
xs = np.array(xs)
xs_rect = np.array(xs_rect)
#%%
s=1
N_w = 4
plt.figure(figsize=(s,s))
A_disp = np.array(W[:N_w, :N_w])/scale_orig
A_disp[np.diag_indices(N_w)] = 0
lim = np.max(np.abs(A_disp))
plt.imshow(A_disp, cmap='RdBu_r', vmin=-lim, vmax=lim)
#xticks are label
plt.xticks(np.arange(N_w), label[:N_w], rotation=-45, fontsize=fontsize_tick)
plt.yticks(np.arange(N_w), label[:N_w], rotation=0, fontsize=fontsize_tick)
#plt.colorbar(label='Synaptic count X sign', )
plt.xlabel('Pre-synaptic neuron', fontsize=fontsize_label)
plt.ylabel('Post-synaptic neuron', fontsize=fontsize_label)
plt.savefig('opp_motion_weights.pdf', bbox_inches='tight', dpi=300)
#%%
s = 1.1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5*s, 1*s), dpi=500)
ylim = 0.6
ax1.set_ylim([-ylim, ylim])
ax1.set_xlabel('Neuron index', fontsize=fontsize_label)
ax1.set_ylabel('Eigenvector loading', fontsize=fontsize_label)
#set tick label sizes

#scatters are open circles
ax1.scatter(range(len(eig_vec[:,ev_ind])), np.real(eig_vec[:,ev_ind]), color='gray', alpha=1, label='Real', marker='o', facecolors='none',s=8, rasterized=True)
#ax1.legend(loc='lower left', fontsize=fontsize_tick,  handletextpad=0.1, handlelength=0.5)
#annotate with eigenvalue
ax1.annotate(r'$\lambda_1$=' f'{((eigenvalues[ev_ind])*scale_orig*0.99):.2f}', xy=(0.1, 0.1), xycoords='axes fraction', fontsize=fontsize_tick)
for i in range(N_w):
    ax1.scatter([top_ind[i]], [np.real(eig_vec[top_ind[i], ev_ind])], color=label_colors_optic[i], alpha=1, marker='o', s=1)


ax1.set_xticks([0, 50000, 100000])
ax1.set_xticklabels(['0', '50,000', '100,000'], fontsize=fontsize_tick)
ax1.set_yticks([-0.5,0,0.5])
ax1.set_yticklabels([-0.5,0,0.5], fontsize=fontsize_tick)


for i in range(N_w):
    ax2.plot(ts, xs[:,i], color=label_colors_optic[i], label=label[i], alpha=0.5, ls='--')
for i in range(N_w):
    ax2.plot(ts, xs_rect[:,i], color=label_colors_optic[i], label=label[i], alpha=1.0, ls='-')
# ax2.legend(loc='upper right', bbox_to_anchor=(1.8, 1), fontsize=fontsize_tick, ncol=2, handletextpad=0.1, 
#                         columnspacing=0.5, title='Linear  Rectified', title_fontsize=fontsize_tick)
ax2.set_xlabel('Time (ms)', fontsize=fontsize_label)
ax2.set_ylabel('Activity (a.u.)', fontsize=fontsize_label)
ax2.set_xticks(np.arange(0, 50 + dt, dt), fontsize=fontsize_tick)
ax2.set_xticklabels(np.arange(0, 50 + dt, dt), fontsize=fontsize_tick)
ax2.set_yticklabels(np.arange(-0.5, 0.75, 0.25), fontsize=fontsize_tick)
ax2.set_xlim([-1, 25])
ax2.set_yticklabels([])
ax2.set_ylim([-ylim, ylim])
fig.tight_layout()
plt.savefig('opp_motion_leg_eigenvector.pdf', bbox_inches='tight', dpi=500)
#%%
A_step = sp.linalg.expm(sp.linalg.logm(W_scale) * 1/(sample_rate*dt))
u_flow = np.zeros_like(x0)
u_right = np.zeros_like(x0)
x0 = np.real(eig_vec[top_ind, 0])
x0 = np.zeros_like(x0)
s = 0.01#stim strength
u_flow[np.array([0,1,2,3])] = s
u_right[np.array([2,3,])] = s
xs_rects = []
for u in ([u_right, u_flow]):
    xs_rect = [x0,]
    xs = [x0,]
    for t in ts[1:]:
        if t>150:#stop stim after 30 ms
            u[...] = 0
        xs_rect.append(A_step @ rectify(xs_rect[-1]+u))
    xs_rects.append(np.real(xs_rect))

s = 1.
plt.figure(figsize=(s*1.5, s))
for j, xs_rect in enumerate(xs_rects):
    xs_rect = np.array(xs_rect)
    for i in range(4):
        plt.plot(ts, xs_rect[:,i], ls=['-', ':'][j], 
                        color=label_colors_optic[i], label=[label, [' ',]*4][j][i], alpha=1)
plt.legend(ncol=2, title='BTF               BTF \nright           left + right', loc=(1.1,0.2), fontsize=fontsize_tick)
plt.xlabel('Time (ms)', fontsize=fontsize_label)
plt.ylabel('Activity (a.u.)', fontsize=fontsize_label)
plt.xticks([0, 100, 200], fontsize=fontsize_tick)
plt.xlim([-10, None])
plt.yticks([0,0.5], fontsize=fontsize_tick)
plt.xticks(fontsize=fontsize_tick)
#plt.ylim(-0.1, 0.6)
plt.savefig('optic_lobe_stim_sim.pdf', bbox_inches='tight', dpi=300)

# %% ring neuron circuit
ev_ind = 44
all_sorted_inds = np.argsort(np.abs(eig_vec[:, ev_ind]))[::-1]
ev_abs = np.abs(eig_vec[all_sorted_inds, ev_ind])
frac_var_ind = np.where(np.cumsum(ev_abs**2)/np.sum(ev_abs**2)>0.75)[0][0]
top_ind = list(all_sorted_inds[:frac_var_ind])
label = list(df_meta_W.loc[top_ind]['hemibrain_type'].values)

x0 = -np.real(eig_vec[top_ind, ev_ind])
W = C_orig[top_ind, :][:, top_ind].todense()*scale_orig
c = 0.63#target amplitude
time_to_c = 0.5#number of steps to reach c
tau = c**(1/time_to_c)
max_abs_eig = np.max(np.abs(np.linalg.eigvals(W)))
scale = (1 - tau)/max_abs_eig
W_scale = scale*W + tau*np.eye(len(top_ind))
W = (W/max_abs_eig) * np.abs(eigenvalues[ev_ind])/np.abs(eigenvalues[0])

#%%
ts =  np.arange(0, T, 1/sample_rate)
A_step = sp.linalg.expm(sp.linalg.logm(W) * 1/(sample_rate*dt))
xs = [x0,]#for recording linear responses
xs_rect = [x0,]#for recording rectified responses
for t in ts[1:]:
    #at each step plug the last time step in and take a 1/sample_rate sized step
    xs.append(A_step @ xs[-1])
    xs_rect.append(A_step @ rectify(xs_rect[-1]))#rectify inputs
xs = np.array(np.real(xs))
xs_rect = np.array(np.real(xs_rect))

#%%
s=1.5
N_w = 10
plt.figure(figsize=(s,s), dpi=400)
A_disp = np.array(W[:N_w, :N_w])/scale_orig
lim = np.max(np.abs(A_disp))
plt.imshow(A_disp, cmap='RdBu_r', vmin=-lim, vmax=lim)
#xticks are label
plt.xticks(np.arange(N_w), label[:N_w], rotation=-45, fontsize=fontsize_tick)
plt.yticks(np.arange(N_w), label[:N_w], rotation=0, fontsize=fontsize_tick)
plt.colorbar(label='Synaptic count X sign')
plt.xlabel('Pre-synaptic neuron', fontsize=fontsize_label)
plt.ylabel('Post-synaptic neuron', fontsize=fontsize_label)
plt.savefig('ring_circuit_weight_matrix.pdf', bbox_inches='tight')
#%%
s = 1.1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5*s, 1*s), dpi=500)
ylim = 0.3
ax1.set_ylim([-ylim, ylim])
ax1.set_xlabel('Neuron index', fontsize=fontsize_label)
ax1.set_ylabel('Eigenvector loading', fontsize=fontsize_label)
#scatters are open circles
ax1.scatter(range(len(eig_vec[:,ev_ind])), -np.real(eig_vec[:,ev_ind]), color='gray', alpha=1, label='Real', marker='o', facecolors='none',s=8, rasterized=True)
#ax1.legend(loc='lower left', fontsize=fontsize_tick,  handletextpad=0.1, handlelength=0.5)
#annotate with eigenvalue
ax1.annotate(r'$\lambda_{45}$=' f'{((eigenvalues[ev_ind])*scale_orig*0.99):.2f}', xy=(0.3, 0.1), xycoords='axes fraction', fontsize=fontsize_tick)
for i in range(N_w):
    ax1.scatter([top_ind[i]], [-np.real(eig_vec[top_ind[i], ev_ind])], color=label_colors_ring[i], alpha=1, marker='o', s=1)
ax1.set_xticks([0, 50000, 100000], fontsize=fontsize_tick)
ax1.set_xticklabels(['0', '50,000', '100,000'], fontsize=fontsize_tick)
ax1.set_yticks([-0.25,0,0.25], fontsize=fontsize_tick)
ax1.set_yticklabels([-0.25,0,0.25], fontsize=fontsize_tick)

xs = np.array(xs)
xs_rect = np.array(xs_rect)
for i in range(N_w):
    ax2.plot(ts, xs[:,i], color=label_colors_ring[i], label=label[i], alpha=0.5, ls='--')
for i in range(N_w):
    ax2.plot(ts, xs_rect[:,i], color=label_colors_ring[i], label=label[i], alpha=1.0, ls='-')
#ax2.legend(loc='upper right', bbox_to_anchor=(1.8, 1), fontsize=fontsize_tick, ncol=2, handletextpad=0.1, 
#                        columnspacing=0.5, title='Linear  Rectified', title_fontsize=fontsize_tick)
ax2.set_xlabel('Time (ms)', fontsize=fontsize_label)
ax2.set_ylabel('Activity (a.u.)', fontsize=fontsize_label)
ax2.set_xticks(np.arange(0, 50 + dt, dt))
ax2.set_yticks([-0.25,0,0.25])
#set xtick label sizes
ax2.set_xticklabels(np.arange(0, 50 + dt, dt), fontsize=fontsize_tick)
ax2.set_xlim([-1, 25])
ax2.set_yticklabels([])
ax2.set_ylim([-ylim, ylim])
fig.tight_layout()
plt.savefig('ring_neuron_eigenvector.pdf', bbox_inches='tight', dpi=500)
#%%
s = 1
angle = np.linspace(0, 0.7, 10)
c = plt.cm.hsv(angle)
T = 200
W_run  = W_scale
#W_run = 0.99*W_run/(np.max(np.abs(np.linalg.eigvals(W_run))))
ts =  np.arange(0, T, 1/sample_rate)
A_step = sp.linalg.expm(sp.linalg.logm(W_run) * 1/(sample_rate*dt))
for stim_neur in [0,4]:
    u = np.zeros(frac_var_ind)
    stim = 0.01
    g = 0.6
    u[...] = 0  
    u[:25] = stim*g
    u[stim_neur] = stim
    xs_rect = [u,]
    xs = [u,]
    for t in ts[1:]:
        if t>150:
            u[...] = 0
        xs_rect.append(A_step @ rectify(xs_rect[-1] + u))
    xs_rect = np.array(xs_rect)
    plt.figure(figsize=(s*1.1,s))
    for i in range(10):
        plt.plot(ts, xs_rect[:,i], ls=['-', ':'][0], color=label_colors_ring[i], alpha=1)
    plt.xlabel('Time (ms)', fontsize=fontsize_label)
    plt.ylabel('Activity (a.u.)', fontsize=fontsize_label)
    plt.xticks([0, 100, 200])
    plt.xlim([-10, None])
    plt.yticks([0,0.5], fontsize=fontsize_tick)
    plt.gca().set_yticklabels([0,0.5], fontsize=fontsize_tick)
    plt.gca().set_xticklabels([0, 100, 200], fontsize=fontsize_tick)
    plt.ylim(-0.1, 0.75)
    #plt.legend([f'{(a*180/np.pi):.1f}' for a in np.linspace(0,np.pi, 9)], loc='upper right', bbox_to_anchor=(1.8, 1),
    #fontsize=7, ncol=1, columnspacing=0.5, title='Degrees visual angle', title_fontsize=7
    #)
    plt.savefig('ring_neuron_stim_' + str(stim_neur)+ '.pdf', bbox_inches='tight', dpi=300)



# %%
