# From connectome to effectome: learning the causal interaction map of the fly brain

## Overall organization of code
Top level directory is: `./conn2eff/` and contains the following subdirectories:


- `data/` 
    - contains large data files and the scripts for slower processing steps. 

- `scripts/`
    - `figs` contains the scripts for generating main figures and data in the paper.
    - `ED_figs` scripts for generate extended data figures.
    
- `src/lib.py` 
    - contains functions that are reused across scripts (simulations and estimators, includes commented out examples).



## Detailed description of how to reproduce results from paper
### Processing raw data
1. Download original data from:
[Flywire](hhttps://codex.flywire.ai/api/download): `neurons.csv`, `connections.csv` and `classification.csv`.
to  folder: `'./data/'` and unzip to csv files.  

2. Run
`'./data/connectome_data.py'`
to convert these csv's to a sparse matrix format and compute its eigendecomposition. Files generated are:
* `C_index_to_rootid.csv`: convert between original ids and index from 0-n neurons in connectome matrix
* `meta_data.csv` : metadata for each neuron when available about cell type etc.
* `connectome_sgn_cnt_sp_mat.npz`: sparse matrix of the connectome
* `connectome_sgn_cnt_scaled_sp_mat.npz` : sparse matrix of the connectome scaled by abs of the largest eigenvalue
* `eigenvalues_1000.npy`: eigenvalues of the connectome
* `eigvec_1000.npy`: eigenvectors of the connectome
* `connectome_sgn_cnt_prob.csv` : connectome in list form with P(excitatory) (used for measurement error sims)

### Reproducing main figures

Go to each folder in figs (fig1, fig2, ...) and first run scripts appended with `data` (if there are any) to generate data files. Then run scripts appended with `fig` to generate the figures, PDFs will be written to the same folder.



### Reproducing extended data figures
#### Robustsness of eigendecomposition results
These include ED fig 9a-d, 10. To generate these figures:
1. Run data script
`'./scripts/ED_figs/eig_robust/eig_robust_data.py'`
to generate the data for the robustness of eigendecomposition results. Files are saved in `./data/`:
* `eigenvalues_robust.nc` and * `eigenvector_robust.nc`: eigenvalues and vectors of the connectome for several different transformations:
    - `original`: the original connectome matrix
    - `tanh_1`: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 1 
    - `tanh_2`: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 2
    - `tanh_10`: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 10
    - `sign`: the connectome matrix binarized by the sign function
    - `shuffled_i`: the connectome matrix with the synapse counts rows and column indices shuffled
    - `measurement_error_i`: the connectome matrix with the synapse counts randomly set by a poisson process with mean equal to original count of each entry and sign randomly flipped according to the prob of NT.
* `eig_corruption.csv`: for SNR=1 the indices and correlations of original to corrupted eigenvectors where corruption is gaussian noise.
One figure is made ED fig 9a and saved in the same folder, `ED_fig_9a_max_correlation_eigenv.png` this gives the correlation between original and corrupted eigenvectors for SNR levels 100,10

To generate all other robustness figures:

2. Run figure script
`'./scripts/ED_figs/eig_robust/eig_robust_fig.py'`

#### Conducance based model simulations
ED Fig 3,4

Run `'./scripts/ED_figs/IV_app_to_conductance_model.py'`


#### Additional AR simulations
ED Fig 5,6

Run `scripts/ED_figs/IV_estimate_ar(p).py` and `scripts/ED_figs/misspec_prior_lin_iv_sim.py`


### Further eigendeomposition analysis
ED Fig 8

Run `'./scripts/ED_figs/eig_further_char.py'` 


