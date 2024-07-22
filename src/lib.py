# %%
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def simulate_sparse_VAR1(W, source_neurons, T=1000, laser_power=1, noise_var=1, seed=None):
    """
    Simulates a sparse VAR1 model.

    Args:
        W (ndarray): The effectome matrix.
        source_neurons (list): A list of neuron indices that are stimulated.
        T (int, optional): Number of time samples to simulate. Defaults to 1000.
        laser_power (float, optional): Laser power. Defaults to 1.
        noise_var (float, optional): Variance of the noise. Defaults to 1.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        tuple: A tuple containing the response matrix (R) and the laser matrix (L).
    """
    if seed is not None:
        np.random.seed(seed)
    n = W.shape[0]
    if isinstance(source_neurons, int):
        source_neurons = [source_neurons]
    elif isinstance(source_neurons, float):
        source_neurons = [int(source_neurons)]

    n_l = len(source_neurons)
    W_lx = np.zeros((n, n_l))  # matrix that maps lasers to neurons
    # could be arbitrary, but for now just set to identity
    for i, source_neuron in enumerate(source_neurons):
        W_lx[source_neuron, i] = 1.
    W_lx = csr_matrix(W_lx)
    L = np.random.randn(n_l, T) * laser_power  # lasers, known b/c we control them
    # preallocate R the response matrix
    R = np.zeros((n, T))
    R[:, 0] = np.random.randn(n) * np.sqrt(noise_var)  # initialize with noise
    # run LDS
    for t in range(1, T):
        # weight prior time step, add laser, add noise
        noise = np.random.randn(n) * np.sqrt(noise_var)
        R[:, t] = W @ R[:, t - 1] + W_lx @ L[:, t] + noise
    return R, L

def suff_stats_fit_prior(X, Y, L, delay=1):
    """
    Calculate sufficient statistics for fitting a prior in IV and IV-bayes estimation.
    closely follows 2-stage least squares described in methods 'Instrumental variable estimator for a linear dynamical system' 
    assumes all variables are zero mean

    Parameters:
    X (ndarray): The stimulated neurons (n_sources, n_samples).
    Y (ndarray): The observed neurons (n_targets, n_samples).
    L (ndarray): The instrumental variable matrix of shape (n_instruments ,n_samples).

    Returns:
    XTX (ndarray): The matrix product of hat_X and its transpose.
    XTY (ndarray): The matrix product of hat_X and Y.
    sig2 (ndarray): The residual variance for IV-bayes estimation.
    hat_W_xy_IV (ndarray): The raw IV estimate.

    """
    X = X[:, :-delay]# source population (stimulated and observed neurons), truncate by delay, so matches Y
    L = L[:, :-delay]# laser power
    Y = Y[:, delay:]# target population (observed neurons), shift by delay
    # generate statistics from which IV and IV-bayes can be calculated
    # regress L on X to give for 1st stage of 2 stage least squares
    hat_W_lx = np.linalg.lstsq(L.T, X.T, rcond=None)[0].T
    hat_X = hat_W_lx @ L # hat_X (just a function of laser) 
    hat_W_xy_IV = np.linalg.lstsq(hat_X.T, Y.T, rcond=None)[0].T # raw IV estimate regress hat_X on Y
    # sufficient statistics to compute IV-bayes estimate
    sig2 = np.var(Y - hat_W_xy_IV @ hat_X, axis=1) # calculate residual variance for estimate of prediction error
    XTY = np.matmul(hat_X, Y.T)
    XTX = hat_X @ hat_X.T

    return XTX, XTY, sig2, hat_W_xy_IV

def fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale=1., prior_var_scale=1., prior_var_constant=1e-16):
    '''IV Bayes estimate of W_xy given sufficient statistics, see suff_stats_fit_prior for source of XTX, XTY, sig2
    see eq 11 in paper for these terms
    XTX (ndarray): The matrix product of hat_X (LSTSQ prediction of X from laser) and its transpose (source x source)
    XTY (ndarray): The matrix product of hat_X and Y,target neurons (source x target).
    sig2 (ndarray): The variance of the observations Y (target).
    a_C (float): The prior mean (target X source).
    prior_mean_scale (float, optional): The scale of the prior mean. Defaults to 1.
    prior_var_scale (float, optional): The scale of the prior variance. Defaults to 1.
    prior_var_constant (float, optional): A constant added to the prior variance. Defaults to 1e-16.

    Returns:
    ndarray: The IV Bayes estimate of W_xy (target x source).

    '''
    # get number of targets
    n_target, n_source = a_C.shape
    prior_mu = a_C*prior_mean_scale#prior mean is proportional to connectome
    gamma2 = (np.abs(prior_mu)+ prior_var_constant)*prior_var_scale#set prior variance proportional to connectome

    #building up eq 10 in paper
    inv_sig2 = 1/sig2
    inv_gamma2 = 1/gamma2
    # the inverse
    # target x source x source
    inv_sig2_XTX = XTX[None, :, :] * inv_sig2[:, None, None]
    diag_inv_gamma2 = np.array([np.diag(inv_gamma2[i]) for i in range(n_target)])
    inv = np.linalg.inv(inv_sig2_XTX + diag_inv_gamma2)

    #the cross covariance, target x source x 1
    inv_sig2_XTY = inv_sig2[:, None] * XTY.T
    inv_gamma2_mu = prior_mu * inv_gamma2
    cross_cov = (inv_sig2_XTY + inv_gamma2_mu)[..., None]
    hat_W_xy_IV_bayes = np.matmul(inv, cross_cov) #matmul vectorizes across all targets
    return hat_W_xy_IV_bayes

# # Test the simulate_sparse_VAR1 function
# n = 5
# W = np.random.randn(n, n)
# W = 0.9 * W / np.abs(np.linalg.eigvals(W)).max()
# source_neurons = np.array([0, 1, 2 ])#source neurons (stimulated by laser and observed)
# target_neurons = np.array([0,1, 3])#target neurons (observed, would typically include source neurons)d
# T = 1000

# laser_power = 1.
# noise_var = 1.
# seed = 42

# R, L = simulate_sparse_VAR1(W, source_neurons, T, laser_power, noise_var, seed)
# # Plot the response matrix
# plt.plot(R.T)
# plt.xlabel('Time')
# plt.ylabel('Neuron')
# plt.show()

# # Test the suff_stats_fit_prior function
# X = R[source_neurons, :]
# Y = R[target_neurons, :]
# L = L
# delay = 1
# XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(X, Y, L, delay)

# n_target_neurons = len(target_neurons)
# n_source_neurons = len(source_neurons)

# # Test the fit_prior_w_suff_stat function
# a_C = W[:, source_neurons][target_neurons, :] # true effectome
# prior_mean_scale = 1.
# prior_var_scale = 1.
# prior_var_constant = 1e-16
# #hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale, prior_var_scale, prior_var_constant)

# prior_mu = a_C*prior_mean_scale#prior mean is proportional to connectome

# hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale, prior_var_scale, prior_var_constant)

# plt.subplot(1, 2, 1)
# for i, source_neuron in enumerate(source_neurons):
#     plt.scatter(W[target_neurons, source_neuron], hat_W_xy_IV[:, i], label=f'Neuron {source_neuron}', alpha=0.5)
# plt.plot([-1, 1], [-1, 1], 'k--')
# plt.xlabel('True weight')
# plt.ylabel('Estimated weight')
# plt.legend(title='Source neuron')
# plt.title('IV')
# plt.axis('square')
# plt.subplot(1, 2, 2)
# for i, source_neuron in enumerate(source_neurons):
#     plt.scatter(W[target_neurons, source_neuron], hat_W_xy_IV_bayes[:, i], label=f'Neuron {source_neuron}', alpha=0.5)
# plt.plot([-1, 1], [-1, 1], 'k--')
# plt.xlabel('True weight')
# plt.ylabel('Estimated weight')
# plt.legend(title='Source neuron')
# plt.title('IV Bayes')
# plt.axis('square')
# plt.tight_layout()