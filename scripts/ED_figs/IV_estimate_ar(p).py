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


def simulate_ARP(W, W_lx, T=1000, laser_power=1, seed=None):
    # W is effectome matrix
    # source_neuron is a list of neuron indices that are stimulated
    # T is number of time samples to simulate
    # n_l is number of lasers
    # seed is random seed
    if seed is not None:
        np.random.seed(seed)
    n = W.shape[0]
    L = np.random.randn(n, T)*laser_power# lasers, known b/c we control them
    #preallocate R the response matrix
    R = np.zeros((n, T))
    R[:, 0] = np.random.randn(n)#initialize with noise
    #run LDS
    for t in (range(1, T)):
        # weight prior time step, add laser, add noise
        noise = np.random.randn(n)
        R[:, t] = W @ R[:, t-1] + W_lx @ L[:, t] + W_lx @ noise
    return R, L

def flatten_weight_matrix(W):
    def flatten_weight_matrix(W):
        """
        Flattens an AR(P) dynamics weight matrix by rearranging its dimensions.
        W = [[W1, W2, ..., WP], [I, 0, 0, ..., 0, 0], [0, I, 0, ..., 0, 0], ..., [0, 0, 0, ... I, 0]]
        Parameters:
        - W (ndarray): The weight matrix to be flattened. It should have shape (D, D, P), where D is the dimension and P is the dynamics order.

        Returns:
        - W_flat (ndarray): The flattened weight matrix with shape (D*P, D*P).
        """
        D, _, P = W.shape
        W_flat = np.zeros((D*P, D*P))
        for i in range(P):
            W_flat[:D, i*D:(i+1)*D] = W[..., i]
        I = np.eye(D*P)
        W_flat[D:, :] = I[:-D, :]
        return W_flat
    D, _, P = W.shape
    W_flat = np.zeros((D*P, D*P))
    for i in range(P):
        W_flat[:D, i*D:(i+1)*D] = W[..., i]
    I = np.eye(D*P)
    W_flat[D:, :] = I[:-D, :]
    return W_flat


D = 5
P_values = [1, 4]
n_samples = np.logspace(np.log10(100), np.log10(10000), 10, dtype=int)
n_repeats = 50  # Number of repeats for each simulation
r2_array = np.zeros((len(n_samples), len(P_values), n_repeats))
q = 0
for j, P in tqdm(enumerate(P_values)):
    for k in range(n_repeats):
        q+=1
        W = np.random.randn(D, D, P)
        W_flat = flatten_weight_matrix(W)
        #scale W_flat to have spectral radius 0.5
        W_flat = 0.5 * W_flat / np.max(np.abs(np.linalg.eigvals(W_flat)))
        W_lx = np.eye(D * P)
        W_lx[D:, D:] = 0
        R, L = simulate_ARP(W_flat, W_lx, T=max(n_samples), laser_power=1, seed=q)
        for i, n in enumerate(n_samples):
            R_sub = R[:D, :n]
            L_sub = L[:D, :n]

            XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(R_sub[:, :], R_sub[:, :], L_sub[:, :])
            W_hat = hat_W_xy_IV
            W_P1 = W_flat[:D, :D]
            R2 = 1 - np.sum((W_hat - W_P1) ** 2) / np.sum(W_P1 ** 2)
            r2_array[i, j, k] = R2

print(r2_array)


#%% plot  results
plt.figure(figsize=(4, 2), dpi=300)
for i, p in enumerate(P_values):
    plt.errorbar(n_samples, np.mean(r2_array[:, i, :], axis=-1), yerr=stats.sem(r2_array[:, i, :], axis=-1), label=f'P={p}')
plt.xlabel('Number of samples')
plt.ylabel(r'$R^2$')
plt.legend(title='Dynamics order')
plt.semilogx()
plt.ylim(-0.1, 1.1)
plt.grid()
# %%
#show estiamted weights and then W_1, W_2, ... W_P in P+1 subplots
plt.figure(figsize=(9, 3), dpi=300)
plt.subplot(1, P+1, 1)
plt.imshow(W_hat, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('IV estimated weights')
plt.xlabel('Pre-synaptic\nneurons')
plt.ylabel('Post-synaptic\nneurons')
plt.xticks([])
plt.yticks([])
for i in range(P):
    plt.subplot(1, P+1, i+2)
    plt.imshow(W_flat[:D, i*D:(i+1)*D], cmap='coolwarm', vmin=-1, vmax=1)
    if i == 0:
        plt.title('True ' + f'$W_{i+1}$')
    else:
        plt.title(f'$W_{i+1}$')

    plt.xticks([])
    plt.yticks([])
# %%
