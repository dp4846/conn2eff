# From connectome to effectome: learning the causal interaction map of the fly brain

## Overall organization of code
Top level directory is: `./conn2eff/` and contains the following subdirectories:

- `data/` 
    - contains large data files and the scripts for slower processing steps. 

- `scripts/` 
    - contains the scripts for generating figures and data in the paper.




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
* `eigenvalues_10000.npy`: eigenvalues of the connectome
* `eigvec_10000.npy`: eigenvectors of the connectome
* `connectome_sgn_cnt_prob.csv` : connectome in list form with P(excitatory) (used for measurement error sims)

...


## Supplementary figures
### Robustsness of eigendecomposition results
1. Run
`'./scripts/eig_robust/eig_robust_data.py'`
to generate the data for the robustness of eigendecomposition results. Files generated are:
* `eigenvalues_robust.nc` and * `eigenvector_robust.nc`: eigenvalues and vectors of the connectome for several different transformations:
    - `original`: the original connectome matrix
    - `tanh_1`: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 1 
    - `tanh_2`: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 2
    - `tanh_10`: the connectome matrix scaled by the max abs syn count then the tanh function applied with a scaling factor of 10
    - `sign`: the connectome matrix binarized by the sign function
    - `shuffled_i`: the connectome matrix with the synapse counts rows and column indices shuffled
    - `measurement_error_i`: the connectome matrix with the synapse counts randomly set by a poisson process with mean equal to original count of each entry and sign randomly flipped according to the prob of NT.
