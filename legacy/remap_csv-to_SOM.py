# tested and running 2026-03-04, modified later
# remap data onto a unsupervised trained SOM classifier, (UMAP and) tSNE, export to FCS
# Changed 2026-06-08 remap to predefined TSNE, running

print ('loading packages and paths...')
import os
import numpy as np
import pandas as pd
import pickle

from flagx.io import FlowDataManager, export_to_fcs
from flagx.gating import SOMClassifier
from flagx.dimred import UMAP
from openTSNE import TSNE

# --- Define path where results are saved to
save_path = './results/workflow_step_wise_unsupervised_inference'
os.makedirs(save_path, exist_ok=True)

# --- Specify where test data is saved and specify the corresponding filenames
# Define path to training data
test_data_path = './data/testing'

# Get list of files in the data directory (only include ones ending with .csv)
test_files = sorted([fn for fn in os.listdir(test_data_path) if fn.endswith('.csv')])

# Load the training files into pandas dataframes
test_data_dfs = [pd.read_csv(os.path.join(test_data_path, fn)) for fn in test_files]

# For each file print the number of channels
# for fn, data_df in zip(test_files, test_data_dfs):
#    print(f'# --- {fn}, number of channels: {data_df.shape[1]}')

# print('\n')

# As an additional check, also print the channel names
# for fn, data_df in zip(test_files, test_data_dfs):
#    print(f'# --- {fn}:\n{data_df.columns.to_list()}')

# print('\n')

# --- Data loading and processing; NOTE: data should be processed the same way as for model training
# Initialize the data manager
fdm = FlowDataManager(
    data_file_names=test_files,
    data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'test_files' list
    data_file_path=test_data_path,
    verbosity=1
)

# Load data into memory
print ('loading data...')
fdm.load_data_files_to_anndata()

# --- Apply preprocessing transformation to each sample, same as for training!
# Example 1: Apply arcsinh with cofactor 150,
# Example 2: Apply log transformation with custom cutoffs
# In both cases, store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
example_1 = True
if example_1:
    preprocessing_kwargs = {'cofactor': 150}
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
else:
    # Define python dictionary mapping channel names to cutoffs (arbitrarily chosen here, adjust if needed)
    channel_name_to_cutoff = {
        'FS INT': 1000, 'SS INT': 800,
        '15-FITC': 300, '13-PE': 300, '34-ECD': 300, '117-PC5.5': 300, '33-PC7': 300,
        '2-APC': 200, '7-APC-AF700': 200, 'HLADR-PB': 200, '45-CO': 200,
    }
    preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
    fdm.sample_wise_preprocessing(
        flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
    )

# NOTE: typically no downsampling at inference time, do so for faster UMAP and t-SNE computation
# --- Downsample each sample to a target number of events
fdm.sample_wise_downsampling(data_set='all', target_num_events=10000)

# Define channels that were used for model training
channels = [
    'FS INT', 'SS INT',
    '15-FITC', '13-PE', '34-ECD', '117-PC5.5', '33-PC7', '2-APC', '7-APC-AF700', 'HLADR-PB', '45-CO'
]
# Extract the processed data matrices from the AnnData objects
data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
# Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
num_events = [x.shape[0] for x in data_matrices]
starting_indices = np.cumsum([0, ] + num_events)
# Concatenate
x_test = np.concatenate(data_matrices, axis=0)

# --- Compute dimensionality reductions on concatenated test samples
# --- SOM
# Load the previously trained SOM model
print ('loading trained SOM model and computing SOM features')
som_clf = SOMClassifier.load(
    filename='som_classifier.pkl',
    filepath='./results/workflow_step_wise_unsupervised_som_training'
)
_, x_som, _, _ = som_clf.transform(x_test)

# --- UMAP
# print ('computing UMAP')
# umap_model = UMAP(n_components=2, n_jobs=-1)
# x_umap = umap_model.fit_transform(x_test)

# --- t-SNE de novo from new data
# print ('computing tSNE')
# tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
# x_tsne = tsne_model.fit(x_test)

# --- t-SNE remapping from stored embedding
tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
tsne_old = pickle.load(open(os.path.join('./results/workflow_step_wise_unsupervised_som_training', 'tsne_embedding.pkl'), 'rb'))
x_tsne = tsne_old.transform(x_test)

# Change back into sample-wise format (input format required by export function)
x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
# x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
# x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

# Export to FCS
print ('exporting to FCS')
export_to_fcs(
    data_list=fdm.anndata_list_,  # Export the test samples
    layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
    sample_wise=False,  # Export one FCS in which the test samples are concatenated
    add_columns=[
        x_soms_1, x_soms_2,
        x_tsnes_1, x_tsnes_2
    ],  # Add columns corresponding to dimensionality reductions, also x_umaps_1, x_umaps_2 possible
    add_columns_names=['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2'],  # added columns, e.g. 'UMAP_1', 'UMAP_2',
    scale_columns=['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2', 'sample_id'],  # Select added columns for scaling
    val_range=(0, 2**20),  # Range to which selected columns are scaled to
    save_path=save_path,
    save_filenames='annotated_test_data.fcs'
)
