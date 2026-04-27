# SOM training (unsupervised) for samples without population annotation. Fast tSNE added. Tested and running 2026-04-27
# shuffle version of the training data used for model fitting. Dimensionality reductions and FCS export based on unshuffled data
# read csv, perform training, dim reduction, export fcs and save model
# Before use, check number and names of channels is consistent across samples (use separate script)
# check if channels like sample_id are included in the csv and need to be scaled for export (if so, add to 'scale_columns')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print('loading packages and paths...')
timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%Hh%M")

from flagx.io import FlowDataManager, export_to_fcs
from flagx.gating import SOMClassifier
from flagx.dimred import UMAP
from openTSNE import TSNE

# --- Define selected Parameters for the workflow ------------------------------------
trainchannels = [
    'FS INT', 'SS INT', '15-FITC', '13-PE', '33-PC7', '2-APC', '7-APC-AF700',
     '34-ECD', '117-PC5.5', 'HLADR-PB', '45-CO'
] # List of channels to be used for training. Check spelling and consistency across samples. Adjust if needed.
# full set for AL1: 'FS INT', 'SS INT', '15-FITC', '13-PE', '33-PC7', '2-APC', '7-APC-AF700', '34-ECD', '117-PC5.5', 'HLADR-PB', '45-CO'
size_per_sample = 80000  # Maximum number of events per sample to be used for model training
SOM_dim = (30, 30)  # Dimensions of the SOM grid. 10x10 for fast testing, 25x25 to 30x30 for better resolution
SOM_epochs = 100 # Number of epochs for SOM training. default 100 for smaller grids, up to 1000

# --- Define path where results are saved to
save_path = './results/workflow_step_wise_unsupervised_som_training'
save_path_data_handling = './results/workflow_step_wise_unsupervised_som_training/data_handling'
os.makedirs(save_path_data_handling, exist_ok=True)

# --- Specify where training data is saved and specify the corresponding filenames
# Define path to training data
training_data_path = './data/training'

# Get list of files in the data directory (only include ones ending with .csv)
training_files = sorted([fn for fn in os.listdir(training_data_path) if fn.endswith('.csv')])

# Load the training files into pandas dataframes
training_data_dfs = [pd.read_csv(os.path.join(training_data_path, fn)) for fn in training_files]

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
# and can be accessed via FDM.sample_sizes_. Since a filename is passed as well, the df is also saved in the
# directory specified via 'save_path_data_handling'.
fdm.check_sample_sizes(filename_sample_sizes_df='sample_sizes.csv')

# Use a built-in plotting function to visualize the number of events per sample.
# Resulting plot also saved to 'save_path_data_handling'.
# fig, ax = plt.subplots(dpi=300)
# fdm.plot_sample_size_df(sample_size_df=fdm.sample_sizes_, ax=ax)
# fig.savefig(os.path.join(save_path_data_handling, 'sample_sizes.png'))
# plt.close(fig)

# --- Apply preprocessing transformation to each sample
# Example 1: Apply arcsinh with cofactor 150,
# Example 2: Apply log transformation with custom cutoffs
# In both cases, store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
example_1 = False # Set to True to apply arcsinh transformation, set to False to apply log transformation with custom cutoffs
if example_1:
    preprocessing_kwargs = {'cofactor': 150}
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
else:
    # Define python dictionary mapping channel names to cutoffs (arbitrarily chosen here, adjust if needed)
    channel_name_to_cutoff = {
        'FS INT': 50000, 'SS INT': 10000, '15-FITC': 100, '13-PE': 300, '33-PC7': 200, '2-APC': 200, '7-APC-AF700': 200, 
         '34-ECD': 200, '117-PC5.5': 200, 'HLADR-PB': 200, '45-CO': 200,
    } #  '-APC-AF750': 200,
    preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
    fdm.sample_wise_preprocessing(
        flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
        )

# --- Downsample each sample to a target number of events
fdm.sample_wise_downsampling(data_set='all', target_num_events=size_per_sample)

# --- Extract concatenated data matrix for model training
# Define channels to be used for model training
channels = trainchannels

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

# Save the trained model
som_clf.save(filename='som_classifier.pkl', filepath=save_path)

# Paul: For in computing dimensionality reductions use unshuffled data for clean data export
_, x_som, _, _ = som_clf.transform(x_train)

time_c = datetime.now()
timesom = time_c - time_b

# --- t-SNE
print ('computing t-SNE...')
tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
# x_tsne = tsne_model.fit_transform(x_train)
x_tsne = tsne_model.fit(x_train)

time_d = datetime.now()
timeSNE = time_d - time_c

# --- UMAP
# print ('computing UMAP...')
# umap_model = UMAP(n_components=2, n_jobs=-1)
# x_umap = umap_model.fit_transform(x_train)

# Change back into sample-wise format (input format required by export function)
x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
# x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
# x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

# Export to FCS
export_to_fcs(
    data_list=fdm.anndata_list_,  # Export the test samples
    layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
    sample_wise=False,  # Export one FCS in which the test samples are concatenated
    add_columns=[
        x_soms_1, x_soms_2,
        x_tsnes_1, x_tsnes_2
    ],  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
    add_columns_names=['SOM_1', 'SOM_2',  'TSNE_1', 'TSNE_2'],  # Add names for added columns
    scale_columns=['SOM_1', 'SOM_2',  'TSNE_1', 'TSNE_2', 'sample_id'],  # Select added columns for scaling
    val_range=(0, 2**20),  # Range to which selected columns are scaled to
    save_path=save_path,
    save_filenames=f'train_w_SOM_{date_time_str}.fcs'
)
timetotal = datetime.now()-timestart
with open(os.path.join(save_path, f'csv_training_{date_time_str}.txt'), 'a') as f:
    f.write(f'"training_files" {date_time_str}: \n')
    for items in training_files:
        f.write(items + "\n")
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"samplesize max": {size_per_sample}\n')
    f.write(f'"SOM_dim": {SOM_dim}\n')
    f.write(f'"SOM_epochs": {SOM_epochs}\n')
    f.write(f'"time data load": {timeload}\n')
    f.write(f'"timesom": {timesom}\n')
    f.write(f'"timeSNE": {timeSNE}\n')
    f.write(f'"timetotal": {timetotal}\n')