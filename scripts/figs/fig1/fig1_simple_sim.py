#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
#set the matplotlib font to arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
cmap = 'bwr'

#%%  FIG 1C example traces of confounded and unconfounded responses
np.random.seed(1)
T = 200 # number of time samples
N = 2 # number of neurons
noise_sd = 1.0
W = np.eye(N) # effectome matrix (no connection between neurons)
eig = np.linalg.svd(W)[1] 
W = W/np.abs(eig[0]) * 0.5 #scale W so that the largest eigenvalue is 0.5
noise = np.random.randn(T, N) * noise_sd
t = np.arange(T) # time vector
period = [2, 3, 5, 7, 10, 16]#add several sine waves of random phase
confound_noise = np.array([np.cos((t/T*2*np.pi + np.random.uniform(0, np.pi*2)) * P) for P in period]).T
confound_noise = confound_noise.sum(1, keepdims=True)
confound_noise /= confound_noise.std()

# run AR simulation
R = np.zeros((N, T))
R_no_confound = np.zeros((N, T)) 
R[:, 0] = noise[0, :]
for t in range(1, T):
    R[:, t] = W @ R[:, t-1] + noise[t, :] + confound_noise[t, :]
    R_no_confound[:, t] = W @ R_no_confound[:, t-1] + noise[t, :]

s = 0.8
T_samps = 100
fig, axs = plt.subplots(3,1, figsize=(3*s,4*s), sharey=False, sharex=True, dpi=200)
colors = ['C2', 'C6', 'C5',]
axs[0].plot(confound_noise[:T_samps], c=colors[0])
axs[0].set_title('Common noise (Z)')
axs[0].set_xticklabels([])

axs[1].plot(R_no_confound[0][:T_samps], c=colors[1])
axs[1].plot(R_no_confound[1][:T_samps], c=colors[2])
axs[1].set_title('Raw responses')

axs[1].set_xticklabels([])

axs[2].plot(R[0][:T_samps], c=colors[1])
axs[2].plot(R[1][:T_samps], c=colors[2])
axs[2].set_title('Confounded responses')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Neural response')
#axs[2].legend(['X', 'Y'], loc=(1.1,0))
axs[0].set_yticklabels([])
axs[2].set_xticks([])
for i in range(3):
    axs[i].set_yticks([])

plt.tight_layout()
plt.savefig('fig1_sim_example.pdf',bbox_inches='tight', transparent=True, pad_inches=0)

#%% FIG 1D  simulation showing IV robustness to confounding
W = np.eye(N) # effectome matrix (no connection between neurons)
W[1, 0] = 0. # add connection from neuron 1 to neuron 0
eig = np.linalg.svd(W)[1] 
W = W/np.abs(eig[0]) * 0.5 #scale W so that the largest eigenvalue is 0.5
Ts = np.logspace(np.log10(1000), np.log10(100000), 5).astype(int)
n_sims = 100 # number of repeats for each simulation of duration T
sim_res = np.zeros((len(Ts), n_sims, 2))
for i, T in enumerate(tqdm(Ts[:])):
    for sim in (range(n_sims)):   
        noise = np.random.randn(T, N) * noise_sd
        t = np.arange(T)
        #confound_noise is a sum of sine waves with random phase
        confound_noise = np.array([np.cos((t/T*2*np.pi + np.random.uniform(0, np.pi*2))*P) for P in period]).T
        confound_noise = confound_noise.sum(1, keepdims=True)
        confound_noise /= confound_noise.std() #unit counfound noise
        #confound_noise[...] = 0
        L = np.random.randn(T) # laser power
        R = np.zeros((N, T)) # number of neurons x number of time samples
        R[:, 0] = noise[0, :]
        for t in range(1, T):
            R[:, t] = W @ R[:, t-1] + noise[t, :] + confound_noise[t, :]
            R[0, t] += L[t] # just add laser to first neuron

        X = R[0, :-1][None, :]# source population (stimulated and observed neurons)
        Y = R[1, 1:][None, :]# target population (observed neurons)
        L = L[None, :-1]
        #LSTQ estimate regress prior time step on next (X_t @ X_t.T)^-1 @ X_t @ Y_{t+1}
        hat_W_xy_lstq = (X @ X.T)**(-1) @ X @ Y.T 
        #IV estimate (X_t @ L_t)^{-1} @ Y_{t+1} @ L_t.T (see eq 5-7)
        hat_W_xy_IV =(X @ L.T)**(-1) @ Y @ L.T 
        sim_res[i, sim, :] = [hat_W_xy_lstq, hat_W_xy_IV]
# %%
s=0.6
plt.figure(figsize=(3*s,3*s), dpi=300)
color = ['C0', 'C1']
for i in range(2):
    plt.errorbar(Ts, sim_res[:,:,i].mean(1), yerr=sim_res[:,:,i].std(1), label=['Least-squares', 'IV'][i], c=color[i])
plt.semilogx()
#truth in dashed black
plt.plot([Ts[0], Ts[-1]], [0,0], c='k', ls='--', label='Ground\ntruth', zorder=1000)
plt.xlabel('# time samples')
plt.ylabel('Estimate')
plt.legend(loc=(1.05,0), fontsize=8)
plt.xlim(5e2,2e5)
plt.title('Effect of X on Y')
#replace scientific notation with regular in xticks
xticks = np.array([1e3, 1e4, 1e5]).astype(int)
#add labels with commas
xtick_labels = [f'{x:,}' for x in xticks]
plt.xticks(xticks, xtick_labels)
plt.savefig('IV_vs_lstq_simple.pdf', bbox_inches='tight', transparent=True, pad_inches=0)

# %% FIG 1A traces of random noise
T = 50
for i in range(2):
    l = np.random.normal(0, 1, T)
    plt.figure(figsize=(1,1), dpi=300)
    plt.plot(l, c='r', lw=0.76)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig('fig1_sim_example_l' +  str(i) +'.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
