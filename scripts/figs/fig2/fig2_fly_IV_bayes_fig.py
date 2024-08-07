#%%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#load data generated from fig2_fly_IV_bayes_data.py
top_dir = '../../../'
data_dir = top_dir + 'data/'
ds = xr.open_dataset(data_dir + '/sim_results.nc')
da_est= xr.open_dataarray( data_dir + '/sim_estimates.nc')

#%% Fig 2 B,D error as function of estimator and number samples for IV and IV-bayes
#single column figure 89 mm,  (89/2)/25.4 = 1.75
s = 1.
sim = 1
lim = 0.04
ticks = np.linspace(-lim, lim, 5)
colors = ['C1', 'C0',]
fontsize_ticks = 6  
fontsize_label = 7 
for i, est in enumerate(['bayes', 'IV']):
    plt.figure(figsize=(s,s))
    y = da_est.sel(sim=sim, T_sub=1000, est=est).values
    plt.scatter(ds['w_true'][:, sim], y, s=3, alpha=0.9, color=colors[i], rasterized=True)
    plt.gca().set_xlim(-lim, lim)
    plt.gca().set_ylim(-lim, lim)
    plt.xticks(ticks, fontsize=fontsize_ticks)
    plt.yticks(ticks, fontsize=fontsize_ticks)
    #set tick labels just to be ends and middle
    tick_labels = [str(ticks[0]), '', '0', '', str(ticks[-1])]
    plt.gca().set_xticklabels(tick_labels)
    plt.gca().set_yticklabels(tick_labels)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('True weights', fontsize=fontsize_label)
    plt.ylabel('Estimated weights', fontsize=fontsize_label)
    #make a diagonal line
    plt.plot([-lim, lim], [-lim, lim], color='k', linewidth=0.5)
    plt.savefig(est + '_wholebrain_example.pdf', bbox_inches='tight', dpi=300, transparent=True)
#%%
s=1.
da_true = ds['w_true']
rss = ((da_true - da_est)**2).sum('w')
tss = (da_true**2).sum('w')
r2 = 1 - rss/tss
rss_mean = rss.mean('sim')
rss_std = rss.std('sim')
x = rss_mean.coords['T_sub']
fig, ax = plt.subplots(1,1, figsize=(s,s), sharex=False, sharey=True)
est_types = ['IV', 'bayes']
tss_av = tss.mean()
colors = ['C0', 'C1']
xticks = [10, 100, 1000, 10000]
ax.loglog(x, rss_mean.sel(est='IV',), label='IV', color=colors[0])
ax.loglog(x, rss_mean.sel(est='bayes',), label='IV-bayes', color=colors[1])
# Plot error bars
ax.errorbar(x, rss_mean.sel(est='IV',), yerr=rss_std.sel(est='IV',), label='IV', color=colors[0], capsize=2, capthick=1, elinewidth=1)
ax.errorbar(x, rss_mean.sel(est='bayes',), yerr=rss_std.sel(est='bayes',), label='IV-bayes', color=colors[1], capsize=2, capthick=1, elinewidth=1)
ax.axhline(tss_av, color='k', linestyle='--', label=r'$R^2=0$')
ax.axhline(tss_av/10, color='k', linestyle=':', label=r'$R^2=0.9$')
ax.set_xticks(xticks, minor=False)
ax.set_yticks([0.01, 1, 100], minor=False)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.set_ylabel('RSS', fontsize=fontsize_label)
ax.set_xlabel('# samples', fontsize=fontsize_label)

plt.savefig('wholebrain_efficiency.pdf', dpi=300, bbox_inches='tight', 
            transparent=True)
#%% Fig 2C illustrative connectome prior example
s=1.4
mu = [0, 8, 0, -4, 0, 0, 0, 0, 5, 0]
var = np.array(np.abs(mu)) + 1
x = np.arange(10)
plt.figure(figsize=(s,s))
plt.errorbar(x, mu, yerr=2*var**0.5, fmt='o', color='k',
              capsize=0.75, capthick=0.75, elinewidth=0.75, markersize=0.8)
plt.grid()
plt.xlabel('Post-synaptic \n  neuron index', fontsize=fontsize_label)
#make y label latex Prior distribution mean (\pm 2\sigma)
#plt.ylabel(r'Prior distribution mean ($\pm 2\sigma$)', fontsize=fontsize_label)
plt.ylabel('Prior mean & SD', fontsize=fontsize_label)
#make symmetric across y axis
plt.gca().set_ylim(-20, 20)
#set tick size
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.tight_layout()
plt.savefig('wholebrain_prior.pdf', bbox_inches='tight', 
            dpi=300, transparent=True)
# %%
