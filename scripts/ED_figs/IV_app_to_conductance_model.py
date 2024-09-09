#%%
import numpy as np
import matplotlib.pyplot as plt
from autograd import jacobian
import autograd.numpy as anp
import matplotlib as mpl
#%% now calculate the estimate using IV method
from tqdm import tqdm
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
label_colors = label_colors[1::2]
#%%
# Nonlinear firing rate function (example: sigmoidal function)
def firing_rate(v, v_start=-50, v_saturate=0, k=0.1, max_rate=100):
    return max_rate / (1 + anp.exp(-k * (v - (v_start + v_saturate) / 2)))
# Differential equation model with noise
def conductance_dvdt(v, delayed_v, laser, input_drive, W, E, v_rest, tau, R, noise_std):
    v = v[:, None]
    delayed_v = delayed_v[:, None]
    laser = laser[:, None]
    input_drive = input_drive[:, None]
    noise = np.random.normal(0, noise_std, size=v.shape)
    dvdt = (R * np.multiply(W, (E - v)) @ firing_rate(delayed_v) + v_rest - v + input_drive + noise + laser) / tau
    return dvdt
def f(v, E, W, R, tau, v_rest, dt):
    v = v[:, None]
    return ((R * anp.multiply(W, (E - v)) @ firing_rate(v) + v_rest - v)/tau*dt).squeeze()

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

def fit_prior_w_suff_stat(XTX, XTY, sig2, a_C, prior_mean_scale=1, prior_var_scale=1):
    #function to take stats from suff_stats_fit_prior and estimate W_xy using prior
    N_stim_neurons = XTX.shape[0]
    prior_W = a_C*prior_mean_scale
    gamma2 = (np.abs(prior_W)+ 1e-16)/prior_var_scale#set prior proportional to connectome
    #eq 5 in paper
    inv_sig2 = 1/sig2
    inv_gamma2 = 1/gamma2
    inv_sig2_XTX = XTX[None, :, :] * inv_sig2[:, None, None]
    inv = np.linalg.inv(inv_sig2_XTX + inv_gamma2[..., None]*np.eye(N_stim_neurons)[None, :, :])
    inv_sig2_XTY = XTY * inv_sig2[:, None, ]
    cov = inv_sig2_XTY + prior_W * inv_gamma2
    hat_W_xy_IV_bayes = np.matmul(inv, cov[..., None]).squeeze()
    return hat_W_xy_IV_bayes


def simulate_conductance_model(W, E=None, I=None, tau=1, R=1, D=5, delay=10, num_steps=1000, x0=0, v_rest=-70, dt=0.1, noise_std=0, laser_power=1):
    '''
    Simulate a conductance-based model of D neurons with Euler's method.

    Parameters:
    W: array-like, shape=(D, D)
        Weight matrix of conductances between neurons.
    E: array-like, shape=(D, D)
        Reversal potentials of the synapses between neurons.
    I: array-like, shape=(num_steps, D)
        Input current to each neuron over time.
    tau: float
        Membrane time constant.
    R: float
        Membrane resistance.
    D: int
        Number of neurons.
    delay: int
        Delay in samples representing axonal conduction delay.
    num_steps: int
        Number of time steps to simulate.
    x0: float
        Initial membrane potential of the neurons.
    v_rest: float
        Resting membrane potential of the neurons.
    dt: float
        Time step for Euler's method.
    noise_std: float
        Standard deviation of the noise added at each time step.
    laser_power: float
        Power of the laser stimulation.

    Returns:
    solution: array, shape=(num_steps, D)
        Membrane potentials of the neurons over time.
    laser: array, shape=(num_steps, D)
        Laser stimulation over time.
    t: array, shape=(num_steps,)
        Time points of the simulation.
        '''
    
    if E is None:#default to excitatory
        E = np.zeros((D, D))#GLUTAMATE receptors reverse potential is ~0mV (excitatory)
        #GABA receptors reverse potential is ~-70mV (inhibitory)
    if I is None:#default to no input
        I = np.zeros((num_steps, D))
    t = np.arange(0, num_steps*dt, dt)
    laser = np.random.normal(0, 1, size=(num_steps, D))*laser_power
    solution = np.zeros((num_steps, D))
    solution[:delay+1, :] = x0
    # Euler's method integration
    for i in tqdm(range(delay, num_steps)):
        dvdt = conductance_dvdt(solution[i-1, :], solution[i-delay, :], laser[i, :], I[i, :], W, E, v_rest, tau, R, noise_std)
        solution[i, :] = solution[i-1, :] + (dvdt * dt).squeeze()
    mu_v = np.mean(solution, axis=0)
    jacobian_f = jacobian(f)
    J = jacobian_f(mu_v, E=np.zeros((D,D)), W=W, R=1, tau=1, v_rest=0, dt=dt)
    return solution, laser, t, J



#%% ED fig 3c,d
np.random.seed(1)
delay = 10
dt = 1.0
tau = 10
num_steps = int(1000*600/dt)# a minute of recording time
D = 5
# Synaptic conductances and reversal potentials
W = np.random.uniform(0.01, 0.02, size=(D, D))
np.fill_diagonal(W, 0)
input_strength = 1.
shift = -20
#these are the two conditions
transient_cut = 200
I_1 = np.ones((num_steps, D))
I_2 = np.ones((num_steps, D))
I_1[:] = shift
I_2[:] = (np.array(np.linspace(50, -50, D)))*input_strength + shift
solutions = []
lasers = []
Js = []
for I in [I_1, I_2]:
    solution, laser, t, J = simulate_conductance_model(W=W, I=I, D=D, delay=delay, tau=tau, 
                                                    num_steps=num_steps, laser_power=2, noise_std=2,dt=dt)
    solutions.append(solution)
    lasers.append(laser)
    #give jacobian of f with respect to v_mean
    mu_v = np.mean(solution, axis=0)
    #jacobian_f = jacobian(f)
    #J = jacobian_f(mu_v, E=np.zeros((D,D)), W=W, R=1, tau=1, v_rest=0, dt=0.1)
    Js.append(J)
solutions = np.array(solutions)

#%%
fontsize_title = 7
fontsize_label = 6
fontsize_tick = 5
s=1
for q, solution in enumerate(solutions):
    plt.figure(figsize=(s*1.1,s), dpi=300)
    for i in range(D):
        plt.plot(t[:-transient_cut]/1000, firing_rate(solution[transient_cut:, i]), label=f'Neuron {i+1}',
                 color=label_colors[i], lw=0.25)
    plt.xlabel('Time (s)', fontsize=fontsize_label)
    plt.ylabel('Firing rate (Hz)', fontsize=fontsize_label)

    #plt.legend(loc=(1.04, 0))
    #plt.grid()
    plt.ylim(0, 100)
    plt.xlim(-1, 20)
    #set tick label size
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    if q==0:
        plt.title('Same average voltage', fontsize=fontsize_title)
    else:
        plt.title('Different average voltage', fontsize=fontsize_title)
    #save fig as pdf 
    plt.savefig(f'./ED_fig_3_example_{q+1}.pdf', bbox_inches='tight')

#%%
for q in range(2):
    solution = solutions[q]
    J = Js[q]
    laser = lasers[q]
    X = solution[transient_cut:].T
    Y = solution[transient_cut:].T
    L = laser[transient_cut:].T
    X = X - np.mean(X, axis=1, keepdims=True)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    L = L - np.mean(L, axis=1, keepdims=True)
    XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(X, Y, L, d=delay)
    hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C=W, prior_mean_scale=1, prior_var_scale=1)

    J_od = np.array([J[i, j] for i in range(D) for j in range(D) if i != j])
    hat_W_xy_IV_od = np.array([hat_W_xy_IV[i, j] for i in range(D) for j in range(D) if i != j])
    hat_W_xy_IV_bayes_od = np.array([hat_W_xy_IV_bayes[i, j] for i in range(D) for j in range(D) if i != j])
    W_od = np.array([W[i, j] for i in range(D) for j in range(D) if i != j])

    x_data = [J_od, W_od]
    y_data = hat_W_xy_IV_od
    x_labels = ['Jacobian evaluated\nat average voltage', r'$W_0$' ' (conductances)']
    y_label = 'IV Estimate'
    s=2.8
    fig, ax = plt.subplots(1, 2, figsize=(s, s/2), sharey=True, dpi=300)
    for i in range(2):
        ax[i].scatter(x_data[i], y_data, color='#1f77b4', 
                      alpha=1,  s=5, marker='o')

        ax[i].set_xlabel(x_labels[i], fontsize=fontsize_label)
        
        m, b = np.polyfit(x_data[i], y_data, 1)
        #include sign of intercept so we dont have + - in the label
        ax[i].axline((0, b), slope=m, color='#ff7f0e', linestyle='--', linewidth=0.5, 
                     label=f'y = {m:.2f}x + {b:.2f}' if b > 0 else f'y = {m:.2f}x - {np.abs(b):.2f}')
        if i ==0:
            ax[i].set_ylabel(y_label, fontsize=fontsize_label)
            #ax[i].set_xticks(np.arange(-0., 0.2, 0.05))
            #ax[i].set_yticks(np.arange(-0., 0.2, 0.05))
            #ax[i].set_xlim(-0.025, 0.15)
            
            #ax[i].set_aspect('equal', adjustable='box')
        #ax[i].set_ylim(-0.025, 0.15)
        ax[i].legend(fontsize=5, loc='upper left')
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].axhline(0, color='gray', linewidth=1, linestyle='--')
        ax[i].axvline(0, color='gray', linewidth=1, linestyle='--')
        #put correlation coefficient in the title
        c = np.corrcoef(x_data[i], y_data)[0, 1]
        ax[i].set_title(f'$r^2 = {c**2:.2f}$', fontsize=fontsize_title)
            #set tick label size

        ax[i].tick_params(axis='x', labelsize=fontsize_tick)
        ax[i].tick_params(axis='y', labelsize=fontsize_tick)

    plt.tight_layout()
    plt.savefig(f'./ED_fig_3cd_est_{q+1}.pdf', bbox_inches='tight')



#%%

plt.figure(figsize=(1,1), dpi=200)
#show the weight matrix in hot cool with colorbar no ticks just label x pre-synaptic and y post-synaptic
v_abs_max = np.max(np.abs(W))
plt.imshow(W, cmap='Reds', vmin=0, vmax=0.02)

plt.xlabel('Pre-synaptic\nneuron', fontsize=fontsize_label)
plt.ylabel('Post-synaptic\nneuron', fontsize=fontsize_label)
plt.title(r'$W_0$', fontsize=fontsize_title)
plt.xticks([])
plt.yticks([])
#only max and min ticks on colorbar
cbar = plt.colorbar(ticks=[0, 0.02], label='Conductance')
cbar.ax.tick_params(labelsize=fontsize_tick)  # Change tick label font size
cbar.set_label('Conductance', fontsize=fontsize_label)
plt.savefig('./ED_fig_3a_W.pdf', bbox_inches='tight')
#%% simulation to compare IV and IV-bayes for sparse matrices
D = 10
# Synaptic conductances and reversal potentials

num_steps = int(1000*1000/dt)# a minute of recording time

input_strength = 10
I = np.ones((num_steps, D))
I[:] = np.linspace(-1, 1, D)*input_strength + 40
#these are the two conditions
transient_cut = 100
solutions = []
Js = []
Ws = []
lasers = []
n_sims = 10
for i in range(n_sims):
    W = np.random.uniform(0.01, 0.02, size=(D, D))
    np.fill_diagonal(W, 0)
    #randomly set 90% of weights to zero
    W[np.random.rand(*W.shape) < 0.9] = 0
    solution, laser, t, J = simulate_conductance_model(W=W, D=D, I=I, delay=delay, num_steps=num_steps, laser_power=2)
    solutions.append(solution)
    Js.append(J)
    Ws.append(W)
    lasers.append(laser)
solutions = np.array(solutions)
#%% plot solution
for q, solution in enumerate(solutions[:1]):
    plt.figure(figsize=(1,1), dpi=300)
    for i in range(D):
        plt.plot(t[:-transient_cut]/1000, firing_rate(solution[transient_cut:, i]), label=f'Neuron {i+1}')
    plt.xlabel('Time (s)', fontsize=fontsize_label)
    plt.ylabel('Firing rate (Hz)', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    #plt.legend(loc=(1.04, 0))
    plt.grid()
    plt.ylim(0, 100)
plt.savefig('./ED_fig_4_firing_rate.pdf', bbox_inches='tight')

#%%
W_od = np.array([W[i, j] for i in range(D) for j in range(D) if i != j])
#make N_pts_subs a log scale from 100 to num_steps
N_pts_subs = np.logspace(2, np.log10(num_steps), 10, dtype=int)
r2s = []
mses = []
for N_pts_sub in tqdm(N_pts_subs):
    r2s_temp = []
    mses_temp = []
    for q, solution in enumerate(solutions):
        J = Js[q]
        W = Ws[q]
        laser = lasers[q]
        L = laser[transient_cut:N_pts_sub].T.copy()
        X = solution[transient_cut:N_pts_sub].T.copy()
        Y = solution[transient_cut:N_pts_sub].T.copy()
        L = L - np.mean(L, axis=1, keepdims=True)
        X = X - np.mean(X, axis=1, keepdims=True)
        Y = Y - np.mean(Y, axis=1, keepdims=True)

        XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(X, Y, L, d=delay)
        hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C=W, prior_mean_scale=1, prior_var_scale=10)

        J_od = np.array([J[i, j] for i in range(D) for j in range(D) if i != j])
        hat_W_xy_IV_bayes_od = np.array([hat_W_xy_IV_bayes[i, j] for i in range(D) for j in range(D) if i != j])
        hat_W_xy_IV_od = np.array([hat_W_xy_IV[i, j] for i in range(D) for j in range(D) if i != j])
        
        r2_IV_bayes = np.corrcoef(hat_W_xy_IV_bayes_od, J_od)[0, 1]**2
        r2_IV = np.corrcoef(hat_W_xy_IV_od, J_od)[0, 1]**2
        #do MSE instead
        mse_IV_bayes = np.mean((hat_W_xy_IV_bayes_od - J_od)**2)
        mse_IV = np.mean((hat_W_xy_IV_od - J_od)**2)
        mses_temp.append([mse_IV, mse_IV_bayes])
        r2s_temp.append([r2_IV, r2_IV_bayes])
    r2s.append(np.array(r2s_temp))
    mses.append(np.array(mses_temp))
r2s = np.array(r2s)
mses = np.array(mses)
print(r2s.shape)
#%%
s = 1 
plt.figure(figsize=(s,s),dpi=200)
start = 3
plt.errorbar(N_pts_subs[start:], r2s[...,0].mean(1)[start:], 
        yerr=r2s[...,0].std(1)[start:],  capsize=3, label='IV', lw=1)
plt.errorbar(N_pts_subs[start:], r2s[...,1].mean(1)[start:], 
        yerr=r2s[...,1].std(1)[start:],  capsize=3, label='IV-bayes', lw=1)
plt.title('Jacobian estimation accuracy', fontsize=fontsize_title)
plt.xlabel('Number of samples', fontsize=fontsize_label)
plt.ylabel('$R^2$', fontsize=fontsize_label)
plt.legend(loc=(1.04, 0), fontsize=fontsize_label)
#set label tick size
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
plt.semilogx()
plt.savefig('./ED_fig_4_r2.pdf', bbox_inches='tight')

#%%
#scatter W_od for IV and IV-bayes
N_pts_sub = 10000
s = 0.6
fig, ax = plt.subplots(1, 2, figsize=(s*4, s*2), sharey=True)
q = 1
solution = solutions[q]
J = Js[q]
W = Ws[q]
laser = lasers[q]
L = laser[transient_cut:N_pts_sub].T
X = solution[transient_cut:N_pts_sub].T
Y = solution[transient_cut:N_pts_sub].T
L = L - np.mean(L, axis=1, keepdims=True)
X = X - np.mean(X, axis=1, keepdims=True)
Y = Y - np.mean(Y, axis=1, keepdims=True)

XTX, XTY, sig2, hat_W_xy_IV = suff_stats_fit_prior(X, Y, L, d=delay)
hat_W_xy_IV_bayes = fit_prior_w_suff_stat(XTX, XTY, sig2, a_C=W, prior_mean_scale=1, prior_var_scale=1)

J_od = np.array([J[i, j] for i in range(D) for j in range(D) if i != j])
hat_W_xy_IV_od = np.array([hat_W_xy_IV[i, j] for i in range(D) for j in range(D) if i != j])
hat_W_xy_IV_bayes_od = np.array([hat_W_xy_IV_bayes[i, j] for i in range(D) for j in range(D) if i != j])
W_od = np.array([W[i, j] for i in range(D) for j in range(D) if i != j])
x_data = [J_od, J_od]
y_data = [hat_W_xy_IV_od, hat_W_xy_IV_bayes_od]
x_labels = ['Jacobian', '']
y_labels = ['IV', 'IV-Bayes']
for i in range(2):
    ax[i].scatter(x_data[i], y_data[i], color=f'C0', marker='o', s=3, label=f'Simulation {q+1}', )

    ax[i].set_xlabel(x_labels[i], fontsize=fontsize_label)
    ax[i].set_ylabel(y_labels[i], fontsize=fontsize_label)
    ax[i].set_xticks(np.arange(-0.2*s, 0.2*s, 0.1*s), fontsize=fontsize_tick)
    ax[i].set_yticks(np.arange(-0.2*s, 0.2*s, 0.1*s), fontsize=fontsize_tick)
    #set tick label size
    ax[i].set_xticks(np.arange(-0.2*s, 0.2*s, 0.1*s))
    ax[i].set_yticks(np.arange(-0.2*s, 0.2*s, 0.1*s))
    ax[i].tick_params(axis='both', labelsize=fontsize_tick)
    ax[i].set_xlim(-0.15*s, 0.15*s)
    ax[i].set_ylim(-0.15*s, 0.15*s)
    ax[i].set_aspect('equal', adjustable='box')
    #ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].axhline(0, color='gray', linewidth=1, linestyle='-')
    ax[i].axvline(0, color='gray', linewidth=1, linestyle='-')
    #put correlation coefficient in the title
    c = np.corrcoef(x_data[i], y_data[i])[0, 1]
    ax[i].set_title(f'$R^2$ = {c**2:.2f}', fontsize=fontsize_title)
plt.tight_layout()
plt.savefig('./ED_fig_4_IV_IV_bayes.pdf', bbox_inches='tight')
# %%

#%% plot jacobian versus W_od
s=0.5
plt.subplots(1, 1, figsize=(5, 2.5), sharey=True)
plt.scatter(J_od, W_od, label='True Jacobian')
plt.xticks(np.arange(-0.2*s, 0.2*s, 0.1*s))
plt.yticks(np.arange(-0.2*s, 0.2*s, 0.1*s))
plt.xlim(-0.15*s, 0.15*s)
plt.ylim(-0.15*s, 0.15*s)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('Jacobian')
plt.ylabel('True Conductance')
#%%calculate correlation between Jacobian and W_od for non-zero weights
rs = []
for q, solution in enumerate(solutions):
    J = Js[q]
    W = Ws[q]
    J_od = np.array([J[i, j] for i in range(D) for j in range(D) if i != j])
    W_od = np.array([W[i, j] for i in range(D) for j in range(D) if i != j])
    J_od_nonzero = J_od[W_od != 0]
    W_od_nonzero = W_od[W_od != 0]
    rs.append(np.corrcoef(J_od_nonzero, W_od_nonzero)[0, 1])
rs = np.array(rs)
print('Correlation between Jacobian and W_od for non-zero weights:', rs.mean())



#%%
#plot of firing rates for each neuron

plt.figure(figsize=(3,2), dpi=150)
for i in range(D):
    plt.plot(t[:-transient_cut]/1000, firing_rate(solution[transient_cut:, i]), label=f'Neuron {i+1}', alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')

#plt.legend(loc=(1.04, 0))
plt.grid()
plt.ylim(0, None)
#plt.xlim(0, None)
plt.show()
#%%

plt.figure(figsize=(4,3), dpi=200)
#show the weight matrix in hot cool with colorbar no ticks just label x pre-synaptic and y post-synaptic
v_abs_max = np.max(np.abs(W))
plt.imshow(W, cmap='Reds', vmin=0, vmax=0.02)

plt.xlabel('Pre-synaptic\nneuron')
plt.ylabel('Post-synaptic\nneuron')
plt.title(r'$W_0$')
plt.xticks([])
plt.yticks([])
#only max and min ticks on colorbar, colorbar should be same height as plot

plt.colorbar(ticks=[0, 0.02], label='Conductance')
plt.tight_layout()


# %%
plt.figure(figsize=(1,1))
v = np.linspace(-100, 100, 1000)
plt.plot(v, firing_rate(v))
plt.xlabel('Membrane Potential (mV)', fontsize=fontsize_label)
plt.ylabel('Firing rate (Hz)', fontsize=fontsize_label)
plt.title('Firing rate function', fontsize=fontsize_title)
#set tick label size
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
plt.savefig('./ED_fig_3_firing_rate.pdf', bbox_inches='tight')
# %%
