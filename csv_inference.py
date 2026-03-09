# workflow_step_wise_supervised_inference
# tested and running 2026-03-05
print('load scripts and data...')
import os
import numpy as np

from flagx.io import FlowDataManager, export_to_fcs
from flagx.gating import SomClassifier, MLPClassifier
from flagx.dimred import UMAP, TSNE

# --- Define path where results are saved to
save_path = './results/workflow_supervised_inference'
os.makedirs(save_path, exist_ok=True)

# Define path to training data
inference_data_path = './data/testing'

# Get list of files in the data directory (only include ones ending with .csv)
inference_files = sorted([fn for fn in os.listdir(inference_data_path) if fn.endswith('.csv')])

# --- Data loading and processing
# Initialize the data manager
fdm = FlowDataManager(
    data_file_names=inference_files,
    data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'training_files' list
    data_file_path=inference_data_path,
    verbosity=1
)

# Load data into memory
fdm.load_data_files_to_anndata()

# --- Apply preprocessing transformation to each sample, same as for training data
# Apply arcsinh with cofactor 150.
# Store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
preprocessing_kwargs = {'cofactor': 150}
fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)

# Define channels that were used for model training
channels = ['FS INT', 'SS INT',
    '15-FITC', '13-PE', '34-ECD', '117-PC5.5', '33-PC7', '2-APC', '7-APC-AF700', 'HLADR-PB', '45-CO']

# Extract the processed data matrices from the AnnData objects
data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
# Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
num_events = [x.shape[0] for x in data_matrices]
starting_indices = np.cumsum([0, ] + num_events)
# Concatenate
x_test = np.concatenate(data_matrices, axis=0)

# Load the previously trained models
som_clf = SomClassifier.load(
    filename='som_classifier.pkl',
    filepath='./results/workflow_step_wise_supervised_training'
)

mlp_clf = MLPClassifier.load(
    filename='mlp_classifier.pkl',
    filepath='./results/workflow_step_wise_supervised_training'
)

# Make prediction for the test data
print('make predictions for test data...')
y_pred_som = som_clf.predict(x_test)
y_pred_mlp = mlp_clf.predict(x_test)

# Change predictions back into sample-wise format (input format required by export function)
y_pred_som = [y_pred_som[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]
y_pred_mlp = [y_pred_mlp[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]


add_columns = [y_pred_som, y_pred_mlp]
add_columns_names = ['pred_som', 'pred_mlp']

# If dimensionality reductions should be computed as well, set: compute_dim_red = True
compute_dim_red = True
if compute_dim_red:

    # --- SOM
    print('compute SOM...')
    _, x_som, _, _ = som_clf.transform(x_test)

    # --- UMAP
    print('compute UMAP...')
    umap_model = UMAP(n_components=2, n_jobs=-1)
    x_umap = umap_model.fit_transform(x_test)

    # --- t-SNE
    print('compute t-SNE...')
    tsne_model = TSNE(n_components=2, n_jobs=-1)
    x_tsne = tsne_model.fit_transform(x_test)

    # Change back into sample-wise format (input format required by export function)
    x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
    x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
    x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

    add_columns += [
        x_soms_1, x_soms_2,
        x_umaps_1, x_umaps_2,
        x_tsnes_1, x_tsnes_2
    ]
    add_columns_names += ['SOM_1', 'SOM_2', 'UMAP_1', 'UMAP_2', 'TSNE_1', 'TSNE_2']

# Export to FCS
export_to_fcs(
    data_list=fdm.anndata_list_,  # Export the test samples
    layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
    sample_wise=False,  # Export one FCS in which the test samples are concatenated
    add_columns=add_columns,  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
    add_columns_names=add_columns_names,  # Add names for added columns
    scale_columns=add_columns_names,  # Select added columns for scaling (all that were added to the file)
    val_range=(0, 2 ** 20),  # Range to which selected columns are scaled to
    save_path=save_path,
    save_filenames='annotated_test_data.fcs'
)