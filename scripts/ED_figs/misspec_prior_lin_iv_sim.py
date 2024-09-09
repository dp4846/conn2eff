#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
#linear IV simulations
def suff_stats_fit_prior(X, Y, L, d=1):
    #generate statistics from which IV and IV-bayes can be calculated
    # regress X on L to give hat_X (just a function of laser) for 1st stage of 2 stage least squares
    hat_W_lx = np.linalg.lstsq(L.T, X.T, rcond=None)[0].T
    hat_X = hat_W_lx @ L 
    hat_W_xy_IV = np.linalg.lstsq(hat_X[:, :-d].T, Y[:,d:].T, rcond=None)[0].T#raw IV estimate
    sig2 = np.var(Y[:, d:] - hat_W_xy_IV@hat_X[:, :-d], axis=1)#calculate residual variance for IV-bayes
    #see eq 5 in paper for where these are used.
    XTY = np.matmul(hat_X[:, :-d], Y[:, d:, None]).squeeze(-1)
    XTX = hat_X @ hat_X.T
    return XTX, XTY, sig2, hat_W_xy_IV

def fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale=1, prior_var_scale=1, prior_constant=0):
    #function to take stats from suff_stats_fit_prior and estimate W_xy using prior
    N_stim_neurons = XTX.shape[0]
    prior_W = a_C*prior_mean_scale
    gamma2 = (np.abs(prior_W)+ prior_constant)/prior_var_scale#set prior proportional to connectome
    #eq 5 in paper
    inv_sig2 = 1/sig2
    inv_gamma2 = 1/gamma2
    inv_sig2_XTX = XTX[None, :, :] * inv_sig2[:, None, None]
    inv = np.linalg.inv(inv_sig2_XTX + inv_gamma2[..., None]*np.eye(N_stim_neurons)[None, :, :])
    inv_sig2_XTY = XTY * inv_sig2[:, None, ]
    cov = inv_sig2_XTY + prior_W * inv_gamma2
    hat_W_xy_IV_bayes = np.matmul(inv, cov[..., None]).squeeze()
    return hat_W_xy_IV_bayes

# simulation showing how estimator is still consistent with a bad prior
def simulate_LDS(W, T=1000, n_l=1, laser_power=1, seed=None):
    # W is effectome matrix
    # source_neuron is a list of neuron indices that are stimulated
    # T is number of time samples to simulate
    # n_l is number of lasers
    # seed is random seed
    if seed is not None:
        np.random.seed(seed)
    n = W.shape[0]
    L = np.random.randn(n_l, T)*laser_power# lasers, known b/c we control them
    #preallocate R the response matrix
    R = np.zeros((n, T))
    R[:, 0] = np.random.randn(n)#initialize with noise
    #run LDS
    for t in (range(1, T)):
        # weight prior time step, add laser, add noise
        noise = np.random.randn(n)
        R[:, t] = W @ R[:, t-1] + L[:, t] + noise
    return R, L
D = 10
tau = 0.1

#plot accuracy of IV-bayes with correct and incorrect prior as a function of number of samples

T = 100000
prior_constants = [1e-5, 1e-4, 1e-3, 1e-2]
n_samples = np.logspace(2, np.log10(T), 10).astype(int)

n_repeats = 10
D = 10  # Assuming the dimensionality D is 10, adjust as necessary

error_correct = np.zeros((len(prior_constants), n_repeats, len(n_samples)))
error_wrong = np.zeros((len(prior_constants), n_repeats, len(n_samples)))
error_naive = np.zeros((len(prior_constants), n_repeats, len(n_samples)))
r2_correct = np.zeros((len(prior_constants), n_repeats, len(n_samples)))
r2_wrong = np.zeros((len(prior_constants), n_repeats, len(n_samples)))
r2_naive = np.zeros((len(prior_constants), n_repeats, len(n_samples)))

for p_idx, prior_const in tqdm(enumerate(prior_constants)):
    for j in range(n_repeats):  
        W = np.random.uniform(0.1, 0.2, size=(D, D))
        prior_mean_wrong = np.random.uniform(0.1, 0.2, size=(D, D))
        # randomly set 90% of weights to zero
        W[np.random.rand(*W.shape) < 0.9] = 0
        prior_mean_wrong[np.random.rand(*W.shape) < 0.9] = 0
        np.fill_diagonal(W, tau)
        prior_mean_correct = W.copy()
        R, L = simulate_LDS(W, T=T, n_l=D, laser_power=1, seed=j)
        for i, n in enumerate(n_samples):
            XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(R[:, :n], R[:, :n], L[:, :n])
            hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C=prior_mean_correct, prior_constant=prior_const)
            hat_W_xy_IV_bayes_wrong = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C=prior_mean_wrong, prior_constant=prior_const)
            error_correct[p_idx, j, i] = np.mean(np.abs(hat_W_xy_IV_bayes - W))
            error_wrong[p_idx, j, i] = np.mean(np.abs(hat_W_xy_IV_bayes_wrong - W))
            error_naive[p_idx, j, i] = np.mean(np.abs(hat_W_xy_IV - W))

            r2_correct[p_idx, j, i] = 1 - np.sum((hat_W_xy_IV_bayes - W)**2)/np.sum((W - np.mean(W))**2)
            r2_wrong[p_idx, j, i] = 1 - np.sum((hat_W_xy_IV_bayes_wrong - W)**2)/np.sum((W - np.mean(W))**2)
            r2_naive[p_idx, j, i] = 1 - np.sum((hat_W_xy_IV - W)**2)/np.sum((W - np.mean(W))**2)



# %%
x_ticks = np.array([1e2, 1e3, 1e4, 1e5]).astype(int)
s=0.8
n_prior = len(prior_constants)
fig, axes = plt.subplots(1, n_prior, figsize=(n_prior * 2.5*s, 4*s), sharex=False, sharey=False)

for p_idx, prior_const in enumerate(prior_constants):
    # Plot R2
    ax = axes[p_idx]
    ax.errorbar(n_samples, np.mean(r2_correct[p_idx], axis=0), yerr=np.std(r2_correct[p_idx], axis=0), label='Correct prior')
    ax.errorbar(n_samples, np.mean(r2_wrong[p_idx], axis=0), yerr=np.std(r2_wrong[p_idx], axis=0), label='Incorrect prior')
    ax.errorbar(n_samples, np.mean(r2_naive[p_idx], axis=0), yerr=np.std(r2_naive[p_idx], axis=0), label='Naive')
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1)
    ax.set_xticks(x_ticks)
    if p_idx == 0:
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('$R^2$')
        ax.set_title(f'Constant added to prior variance \n {prior_const:.0e}')
        
    elif p_idx == n_prior - 1:
        ax.legend(loc=(1.05, 0.5))
        ax.set_title(f'{prior_const:.0e}')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(f'{prior_const:.0e}')
        
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
#%% plot example correct and incorrect priors in 2,1 plot
fig, axes = plt.subplots(1,3, figsize=(6, 8))
ax = axes[0]
#show w the true effectome
ax.imshow(W, cmap='Reds')
#remove ticks
ax.set_xticks([]); ax.set_yticks([])
#x and y labels pre-synaptic and post-synaptic neurons
ax.set_xlabel('Pre-synaptic neurons')
ax.set_ylabel('Post-synaptic neurons')
ax.set_title('True effectome')

ax = axes[1]
ax.imshow(prior_mean_correct, cmap='Reds')
#remove ticks
ax.set_xticks([]); ax.set_yticks([])
#x and y labels pre-synaptic and post-synaptic neurons

ax.set_title('Correct prior')
ax = axes[2]
ax.imshow(prior_mean_wrong, cmap='Reds')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Incorrect prior')
# %%

# %%
