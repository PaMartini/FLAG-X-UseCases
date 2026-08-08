# SOM training (unsupervised) for samples without population annotation. Fast tSNE and automated clustering (PARC) added. 
# shuffle version of the training data used for model fitting. Dimensionality reductions and FCS export based on unshuffled data
# read FCS or csv files, perform training, dim reduction, export FCS and save models (optional)
# Before use, check number and names of channels is consistent across samples (use separate script)
# if channels like sample_idx or population1 (from manual sampling) are included in the FCS or csv data 
  # they need to be scaled for export (if so, add to 'scale_columns')
# Sample_id channel is added by script to tag the different files
# config parameters drawn from yaml files, select and configure suitable file 
# works with FCS and csv (english version) files, tested and running 2026-07-30

import os
import yaml
import numpy as np
from datetime import datetime

print('loading packages and paths...')
timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%Hh%M")

import pickle
import parc
from flagx.io import FlowDataManager, export_to_fcs
from flagx.gating import SOMClassifier
# from flagx.dimred import UMAP
from openTSNE import TSNE

# --- selected Parameters for the workflow are drawn from YAML files, select and configure suitable file-------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_AL1.yml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
trainchannels = config.get('trainchannels')
size_per_sample = config.get('size_per_sample')  # Maximum number of events per sample to be used for model training
SOM_dim = tuple(config.get('SOM_dim'))  # Dimensions of the SOM grid. 10x10 for fast testing, 25x25 to 30x30 for better resolution
SOM_epochs = config.get('SOM_epochs')  # Number of epochs for SOM training. default 100 for smaller grids, up to 1000
val_range_list = config.get('val_range') 
val_range = tuple(val_range_list)  
trafo_arcsinh = config.get('trafo_arcsinh')
arcsinh_div = config.get('arcsinh_div')
channel_name_to_cutoff = config.get('channel_name_to_cutoff')
lin_trafo_FSSS = config.get('lin_trafo_FSSS')
calcTSNE = config.get('calcTSNE')

# --- validate function: configured channel names must be present in each loaded input file
def validate_training_channels(adata_list, configured_channels):
    if not adata_list:
        raise ValueError('No AnnData objects were loaded from the input files.')

    for channel in configured_channels:
        available_channels = list(adata_list[0].var_names)
        if channel not in available_channels:
            raise KeyError(
                f"Training channel '{channel}' could not be found. "
                f"Available channels: {available_channels}"
            )

    for adata in adata_list:
        for channel in configured_channels:
            if channel not in adata.var_names:
                raise KeyError(
                    f"Channel '{channel}' is missing from one of the input files. "
                    f"Available channels: {list(adata.var_names)}"
                )

    return list(configured_channels)

# --- Define path where results are saved to
save_path = './results/workflow_step_wise_unsupervised_som_training'
save_path_data_handling = './results/workflow_step_wise_unsupervised_som_training/data_handling'
os.makedirs(save_path_data_handling, exist_ok=True)

# --- Specify where training data is saved and specify the corresponding filenames
# Define path to training data
training_data_path = './data/training'

# Get list of flow cytometry files in the data directory (prefer .fcs files, fall back to .csv for compatibility)
training_files = sorted([
    fn for fn in os.listdir(training_data_path)
    if fn.lower().endswith(('.fcs', '.csv'))
])

if not training_files:
    raise FileNotFoundError(f'No training files found in {training_data_path}')

# --- Data loading and processing
# Initialize the data manager
fdm = FlowDataManager(
    data_file_names=training_files,
    data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'training_files' list
    data_file_path=training_data_path,
    save_path=save_path_data_handling,
    verbosity=1
    )
    
# Load data into memory
print ('loading data...')
# Data samples are now stored not in a Pandas DataFrames, but in a list of AnnData object.
# This list is an attribute of the FlowDataManager class and can be accessed via FlowDataManager.anndata_list_.
# AnnData is a Python class similar to Pandas DataFrames but with more options and functions for data annotation.
# see: https://anndata.readthedocs.io/en/stable/
fdm.load_data_files_to_anndata()

# --- Check the number of events per sample
# Create a dataframe with the columns sample and n_events, this df is an attribute of the FDM instance
# and can be accessed via FDM.sample_sizes_. The df is stored as a global variable 'sample_sizes_df'.
fdm.check_sample_sizes()
sample_sizes_df = fdm.sample_sizes_

# --- Apply preprocessing transformation to each sample
# store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
trafo_arcsinh = trafo_arcsinh # (from YAML, True = arcsinh transformation, False = log transformation with custom cutoffs)
if trafo_arcsinh:
    preprocessing_kwargs = {'cofactor': arcsinh_div}
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
else:
    channel_name_to_cutoff = channel_name_to_cutoff  # This dictionary is defined in the config YAML file
    preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
    fdm.sample_wise_preprocessing(
        flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
        )

# Optional: 'FS INT' and 'SS INT' will be transformed by division (see YAML config)     
if lin_trafo_FSSS:
    for adata in fdm.anndata_list_:
        if 'FS INT' in adata.var_names:
            adata[:, 'FS INT'].X = adata[:, 'FS INT'].X / 300000
        if 'SS INT' in adata.var_names:
            adata[:, 'SS INT'].X = adata[:, 'SS INT'].X / 300000

# --- Downsample each sample to a target number of events
fdm.sample_wise_downsampling(data_set='all', target_num_events=size_per_sample)

# --- Extract concatenated data matrix for model training
# Define channels to be used for model training
channels = validate_training_channels(fdm.anndata_list_, trainchannels)

# Extract the processed data matrices from the AnnData objects
data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
# Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
num_events = [x.shape[0] for x in data_matrices]
starting_indices = np.cumsum([0, ] + num_events)
# Concatenate
x_train = np.concatenate(data_matrices, axis=0)

# Generate shuffle version of the training data
idx_shuffle = np.random.permutation(x_train.shape[0])
x_train_shuffled = x_train[idx_shuffle, :].copy()

time_b = datetime.now()
timeload = time_b - timestart

# --- SOM training
print('training SOM model...')
# Instantiate the SOMClassifier, set hyperparameters
som_clf = SOMClassifier(
    som_topology='planar',
    som_grid_type='rectangular',
    som_dimensions=SOM_dim,
    neighborhood='gaussian',
    gaussian_neighborhood_sigma=0.1,
    initialization='pca',
    n_epochs=SOM_epochs,
    radius_0=-0.25,
    radius_n=0.1,
    radius_cooling='exponential',
    learning_rate_0=0.1,
    learning_rate_n=0.001,
    learning_rate_decay='exponential',
    unlabeled_label=-999,
    verbosity=1
)

# Model fitting. Since no labels are available only the SOM component is trained in an unsupervised fashion.
# Still, a dummy label vector must be passed containing the label chosen to indicate unlabeled events
y_dummy = np.ones(x_train.shape[0]) * -999

# Paul: Use shuffled training data for model fitting
som_clf.fit(X=x_train_shuffled, y=y_dummy)

# Save the trained model (optional)
# som_clf.save(filename='som_classifier.pkl', filepath=save_path)

# Paul: For in computing dimensionality reductions use unshuffled data for clean data export
_, x_som, _, _ = som_clf.transform(x_train)

time_c = datetime.now()
timesom = time_c - time_b

# --- t-SNE
if calcTSNE:
    print ('computing t-SNE...')
    tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
    x_tsne = tsne_model.fit(x_train)
    # pickle.dump(x_tsne, open(os.path.join(save_path, 'tsne_embedding.pkl'), 'wb'))
time_d = datetime.now()
timeSNE = time_d - time_c

# --- UMAP
# print ('computing UMAP...')
# umap_model = UMAP(n_components=2, n_jobs=-1)  
# x_umap = umap_model.fit_transform(x_train)

# --- PARC clustering
print('computing PARC clustering...')
Parc1 = parc.PARC(x_train, jac_std_global=0.15)
Parc1.run_PARC() # run the clustering
parc_labels = Parc1.labels

# Change back into sample-wise format (input format required by export function)
x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
# x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
# x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
if calcTSNE:
    x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
parc_1 = [parc_labels[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]

# Export to FCS
if calcTSNE:
    add_columns = [
        x_soms_1, x_soms_2,
        x_tsnes_1, x_tsnes_2,
        parc_1
    ]
    add_columns_names = ['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2', 'PARC_labels']
    scale_columns = ['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2', 'PARC_labels']
else:
    add_columns = [
        x_soms_1, x_soms_2,
        parc_1
    ]
    add_columns_names = ['SOM_1', 'SOM_2', 'PARC_labels']
    scale_columns = ['SOM_1', 'SOM_2', 'PARC_labels']

export_to_fcs(
    data_list=fdm.anndata_list_,  # Export the test samples
    layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
    sample_wise=False,  # Export one FCS in which the test samples are concatenated
    add_columns=add_columns,
    add_columns_names=add_columns_names,
    scale_columns=scale_columns,
    val_range=val_range,
    save_path=save_path,
    save_filenames=f'train_unsup_{date_time_str}.fcs'
)
timetotal = datetime.now()-timestart
with open(os.path.join(save_path, f'fcs_training_{date_time_str}.txt'), 'a') as f:
    f.write(f'"training_files and cell numbers" {date_time_str} \n')
    for index, row in sample_sizes_df.iterrows():
        f.write(f'"{row["sample"]}": {row["n_events"]}\n')
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"size max per sample": {size_per_sample}\n')
    f.write(f'"trafo_arcsinh": {trafo_arcsinh} "arcsinh cofactor": {arcsinh_div}\n')
    f.write(f'"channel cutoff for log trafo": {channel_name_to_cutoff}\n')
    f.write(f'"lin_trafo_FSSS": {lin_trafo_FSSS}\n')
    f.write(f'"SOM_dim": {SOM_dim}\n')
    f.write(f'"SOM_epochs": {SOM_epochs}\n')
    f.write(f'"val_range": {val_range}\n')
    f.write(f'"time data load": {timeload}\n')
    f.write(f'"timesom": {timesom}\n')
    f.write(f'"timeSNE": {timeSNE}\n')
    f.write(f'"timetotal": {timetotal}\n')