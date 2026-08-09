# train models on labelled csv files, export fcs file with predictions and dimensionality reduction
# save models for prediction on new data.
# "shuffle" set to False in data loader, but added manually before model training. 
# expects column 'population' which is used as label for supervised training.
# csv files are concatenated and tagged in a new column 'sample_id' by the script
# expects column 'sample_idx' which is used to identify samples in the data matrix concatenated beforehand.
# before run: Check number and names of channels is consistent across samples (use separate script)
# optional: linear transformation for FS INT and SS INT channels 
# provide parameters in yaml file. tested and running 2026-07-31

print('loading packages and paths...')
import os
import yaml
import numpy as np
import pandas as pd
import pickle
import torch
from datetime import datetime

timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%H-%M")

from flagx.io import FlowDataManager, export_to_fcs
from flagx.gating import SOMClassifier, MLPClassifier
# from flagx.dimred import UMAP
from openTSNE import TSNE

# --- selected Parameters for the workflow are drawn from YAML files, select and configure suitable file-------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Bcell.yml')
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
# calcTSNE = config.get('calcTSNE') # currently TSNE always calculated

# --- Define path where results are saved to
save_path = './results/workflow_step_wise_supervised_training'
save_path_data_handling = './results/workflow_step_wise_supervised_training/data_handling'
os.makedirs(save_path_data_handling, exist_ok=True)

# --- Specify where training data is saved and specify the corresponding filenames
# Define path to training data
training_data_path = './data/training_supervised'

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

# --- Apply preprocessing transformation to each sample
# trafo_ash: Apply arcsinh with cofactor 
# trafo_log: Apply log transformation with custom cutoffs
# In both cases, store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
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
# Set target_num_events to 1000 for fast model training in this example
fdm.sample_wise_downsampling(data_set='all', target_num_events=size_per_sample)

# --- Extract concatenated data matrix for model training
# Define channels to be used for model training
channels = trainchannels

# Extract the processed data matrices from the AnnData objects
data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
# Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
num_events = [x.shape[0] for x in data_matrices]
starting_indices = np.cumsum([0, ] + num_events)

# Use dataloader with batchsize -1 (= all data) to extract the transformed data matrix and label vector
# Note that the data matrix from which we retrieve the labels is the original non-transformed data (label_layer_key='no_trafo').
# Otherwise, we would get the arcsinh-transformed labels
data_loader = fdm.get_data_loader(
    data_set='all',
    channels=channels,
    label_key='population',
    label_layer_key='no_trafo',
    shuffle=False,
    batch_size=-1,
)
x_train, y_train = next(iter(data_loader))

# Paul: shuffle=False, shuffle manually; START
idx_shuffle = np.random.permutation(x_train.shape[0])
x_train_shuffled = x_train[idx_shuffle, :].copy()
y_train_shuffled = y_train[idx_shuffle].copy()
# Paul; END

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
    n_epochs=SOM_epochs,  # 1000,
    radius_0=-0.25,
    radius_n=0.1,
    radius_cooling='exponential',
    learning_rate_0=0.1,
    learning_rate_n=0.001,
    learning_rate_decay='exponential',
    # unlabeled_label=-999,
    verbosity=1
)

# Train SOM classifier in supervised fashion, use shuffled data
som_clf.fit(X=x_train_shuffled, y=y_train_shuffled)

# Save the trained model
som_clf.save(filename=f'som_classifier_{date_time_str}.pkl', filepath=save_path)

_, x_som, _, _ = som_clf.transform(x_train)

time_c = datetime.now()
timesom = time_c - time_b

# Instantiate the MLP, set hyperparameters
mlp_clf = MLPClassifier(
    layer_sizes=(128, 64, 32),
    n_epochs=20,
    data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 1},
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbosity=2
)

# Paul: Train MLP classifier in supervised fashion, use shuffled data
mlp_clf.fit(X=x_train_shuffled, y=y_train_shuffled)

# Save the trained model
mlp_clf.save(filename=f'mlp_classifier_{date_time_str}.pkl', filepath=save_path)

# Paul: Predict labels for the training data. Use the unshuffled version of the data
y_pred_som = som_clf.predict(x_train)
y_pred_mlp = mlp_clf.predict(x_train)

time_d = datetime.now()
timemlp = time_d - time_c

# --- t-SNE training, save embedding (optional)
print ('computing t-SNE...')
tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
x_tsne = tsne_model.fit(x_train)
# pickle.dump(x_tsne, open(os.path.join(save_path, 'tsne_embedding.pkl'), 'wb'))

time_e = datetime.now()
timeSNE = time_e - time_d

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

# Paul: Also change format of predictions
y_preds_som = [y_pred_som[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]
y_preds_mlp = [y_pred_mlp[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]

# Paul: Add predictions for training data to be exported to FCS
export_to_fcs(
    data_list=fdm.anndata_list_,  # Export the test samples
    layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
    sample_wise=False,  # Export one FCS in which the test samples are concatenated
    add_columns=[
        x_soms_1, x_soms_2,
        x_tsnes_1, x_tsnes_2,
        y_preds_som, y_preds_mlp
    ],  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
    add_columns_names=['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2', 'y_pred_som', 'y_pred_mlp'],  # Add names for added columns
    scale_columns=['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2', 'y_pred_som', 
                   'y_pred_mlp', 'sample_idx', 'population'],  #  Select added columns for scaling
    val_range=val_range,  # Range to which selected columns are scaled to
    save_path=save_path,
    save_filenames=f'data_supervised_training_{date_time_str}.fcs'
)
timetotal = datetime.now()-timestart
with open(os.path.join(save_path, f'csv_supervised_training_{date_time_str}.txt'), 'a') as f:
    f.write(f'"training_files" {date_time_str}: \n')
    for items in training_files:
        f.write(items + "\n")
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"total events": {x_train.shape[0]}\n')
    f.write(f'"number of populations for training": {int(np.max(y_train))}\n')
    f.write(f'"samplesize limited to": {size_per_sample}\n')
    f.write(f'"trafo_arcsinh": {trafo_arcsinh} "arcsinh cofactor": {arcsinh_div}\n')
    f.write(f'"channel cutoff for log trafo": {channel_name_to_cutoff}\n')
    f.write(f'"lin_trafo_FSSS": {lin_trafo_FSSS}\n')
    f.write(f'"SOM_dim": {SOM_dim}\n')
    f.write(f'"SOM_epochs": {SOM_epochs}\n')
    f.write(f'"val_range": {val_range}\n')
    f.write(f'"time data load": {timeload}\n')
    f.write(f'"timesom": {timesom}\n')
    f.write(f'"timemlp": {timemlp}\n')
    f.write(f'"timeSNE": {timeSNE}\n')
    f.write(f'"timetotal": {timetotal}\n')
