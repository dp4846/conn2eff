# Code: From connectome to effectome=:From connectome to effectome: learning the causal interaction map of the fly brain

## Overall organization of code
Top level directory is: `./conn2eff/` and contains the following subdirectories:

- `data/` 
    - contains large data files and the scripts for slower processing steps. 

- `scripts/` 
    - contains the scripts for generating figures and data in the paper.




## Detailed description of how to reproduce results from paper
### Processing raw data
1. Download original data from:
[Flywire](hhttps://codex.flywire.ai/api/download):neurons.csv connections.csv classification.csv
to  folder: `'./data/'` and unzip.  

2. Run
`'./data/convert_stringer_mat_to_xr.py'`
to convert from the .mat format to xarray dataset format.
    -  Files are saved to: `'./stringer_2019/processed_data/neural_responses/'` prepended with:
        -   raw_ is just the raw response
        -   spont_sub_ is the mean subtracted responses with spontaneous activity subtracted (see Stringer 2019)
        All recordings including cell position and 3 recordings have tdtomato+ labels saved in a csv file in
`'./stringer_2019/processed_data/red_cell/'`

### Estimating signal and noise covariance matrices
For simulations we need estimates of signal and noise covariance from the original data, run: `'./data/est_mean_SN_cov_stringer.py'` saves to: `'./processed_data/stringer_sn_covs/'`. (~4 hour to run)

### Figures

#### Figure 1: 
Conceptual figure describing issues involved in estimating eigenspectra. No data files needed.

#### Figure 3:
Estimates of power-laws from neural data (this has to be run before Figure 2 scripts because those simulations are based on estimates of the eigenspectra of the neural data).
1. `'figures/fig3/est_sig_eig_spec_data.py'`
first calculates eigenmoments and bootstrap samples of eigenmoments
saved in folder respectively as 
`'./mom_est.nc'` and `'./mom_dist.nc'` (~1 hour to run).
then estimates power laws using cvPCA and power laws and broken powerlaws using MEME. Saves these in csv
`'str_pt_estimates_all.csv'`
2. Then for confidence intervals we use 
`'bpl_ci_data.py'`
that saves to:
`'bpl2_sims'` (a day or so to run)
then used in `'est_sig_eig_spec_figs.py'`
3. `'single_neuron_lin_rf_data.py'` estimates single neuron linear RFs. These are saved to: `'./stringer_2019/processed_data/eig_tuning/'` prepended with `'neur_rf_est_'`.


#### Figure 2:
Multivariate normal simulations from true power-laws.
1. `'small_power_law_sims_'` run small simulations (~100 neurons) to demonstrate differences between MEME and cvPCA saves to: `'small_power_law_sims_params.csv'` and `'small_power_law_sims.nc'`  respectively the parameters of the fits and the eigenspectra of ground truth and estimates.
2. `'match_stringer_cov_sims_data'` run simulations matching signal and noise covariance from original data saves files to `'meme_vs_cvpca_pl_est_match_str_sim'` (~24 hours to run).



#### Figure 4:

1. `'eigenmode_tuning_data.py'` generates the data that is the basis of all Figure 4. Because these files are large they are saved to: 
`'./stringer_2019/processed_data/eig_tuning/'`. These are analyses of the 7 natural image response recordings, they are prepended with the following strings:
    - `'sp_cov_stim_u_r_` and `'sp_cov_neur_u_r_` for singular vectors of the estimated stimulus and neural signal response covariance matrices respectively.
    - `'eig_rf_est_`, `'eig_lin_resp_'`, `'eig_pc_r2_est_'` for the top N eigenmodes these are the estimated linear RF weights, the responses of these filters, and R2 values of their predictions.
(~30 minutes to run)

2. `'eigenmode_tuning_figs.py'` generates all the figures in Figure 4. 


#### Figure 5:
1. `vis_phys_geom_data.py` calculates SNR and R2 performance of gabors and linear filters for all neurons and top eigenmode tuning curves saves to `'./stringer_2019/processed_data/eig_tuning/'` prepended with:
    - `'neur_snr_'` and `'pc_snr_'` for SNR of neurons and PCs respectively for each recordings.
    - R2 for linear model across a variety of conditions (image size, variance stabilization, dimensionality reduction of images):
        - `'neur_pc_r2_'` for each recordings gives noise corrected R2 for every neuron (some are unstable for more accurate estimates filter by SNR>0.1). 
        - `'eig_pc_r2'` is all recordings top N eigenmodes R2 but not noise corrected b/c wouldn't make a difference since these are assumed underlying signal.
    - R2 for gabor energy model across a few conditions (image size, variance stabilization):
        - `'neur_gabor_r2_'` for each recordings gives noise corrected R2 for every neuron (some are unstable for more accurate estimates filter by SNR>0.1). 
        - `'eig_gabor_r2'` is all recordings top N eigenmodes R2 but not noise corrected.
Also, plots example gabors for visualization `'example_gabor_filters.pdf'`.
(~30 minutes to run)
2. `vis_phys_geom_figs.py` generates all the figures in Figure 5 and supplementary figures.



### GENERAL TODO

I am currently just making sure all figures can be made and noting where data should be coming from and putting it all into _data files. text with a check box is a todo item.
- [ ] run everything from scratch
    - [ ] bpl_ci_data
    - [ ] match_stringer_cov_sims_data
- [ ] change receptive field names for vis_phys_geom_data and eigenmode_tuning_data to raw_
- [x] update code to use data folder
- [x] check that using raw neural data works
- [x] put all definitions into src
- [x] put RFs and directories into src
- [x] use correct mean subtraction method.
- [x] put src test into v1_bpl_sims.
- [X] run bpl_ci to make sure working for 2 power laws.
- [X] replace figure 5 with new figure.
- [X] supplementary figure of population SNR differences (just used summary stats).
- [x] supplementary figure of gabor regression for 
    - [X] neurons
    - [X] PCs.
    - [X] check if using powers of PC's can give gabor complex and simple cells (no).
- [X] redo fig 3a-d with new figures
- [X] redo fig 4b-d with new figures

- [X] use participation ratio.

- [X] get recording X corresponding to actual id.

