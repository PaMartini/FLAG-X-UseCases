# workflow_step_wise_supervised_inference
# calculates population statistics and exports fcs with population predictions and optional dim reduction. 
# populations to analyze can be selected in config_Sysmex.yml
# for mere caluclation of population statistics, set compute_dim_red = False
# runs with fcs or csv (english format) input files, several files at a time 
# check that preprocessing (trafo)and channels are identical to training samples
# path to pkl trained models is defined in line 130 etc (currently ./data/models)
# TSNE inference to trained TSNE model included as an option, working, but not very well.
# tested and running 2026-08-02

print('loading scripts and data...')
import os
import numpy as np
from datetime import datetime
import pickle
import pandas as pd
import yaml

timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%H-%M")

from flagx.io import FlowDataManager, export_to_fcs
from flagx.gating import SOMClassifier, MLPClassifier
from flagx.dimred import UMAP
from openTSNE import TSNE
import anndata as ad

# --- selected Parameters from YAML files, select and configure suitable file, parameters identical to training!-------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Sysmex.yml')
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
calcTSNE = config.get('calcTSNE')
calcchannels = config.get('calcchannels') # List of channels for which to calculate median intensities per population
calc_populations = config.get('calc_populations')  # Optional list of predicted populations for which to compute channel medians
compute_dim_red = config.get('compute_dim_red')  

# --- Define path where results are saved to
save_path = './results/workflow_supervised_inference'
os.makedirs(save_path, exist_ok=True)

# Define path to inference data
inference_data_path = './data/testing'

# Get list of flow cytometry files in the data directory (prefer .fcs files, fall back to .csv for compatibility)
inference_files = sorted([
    fn for fn in os.listdir(inference_data_path)
    if fn.lower().endswith(('.fcs', '.csv'))
])
if not inference_files:
    raise FileNotFoundError(f'No inference files found in {inference_data_path}')

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
# trafo_arcsinh: Apply arcsinh with cofactor 150,
# trafo_log: Apply log transformation with custom cutoffs
# In both cases, store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
if trafo_arcsinh:
    preprocessing_kwargs = {'cofactor': arcsinh_div}
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
else:
    # Define python dictionary mapping channel names to cutoffs (arbitrarily chosen here, adjust if needed)
    channel_name_to_cutoff = {
         'FS INT': 50000, 'SS INT': 10000, '15-FITC': 100, '13-PE': 300, '33-PC7': 200, '2-APC': 200, '7-APC-AF700': 200, 
         '34-ECD': 200, '117-PC5.5': 200, 'HLADR-PB': 200, '45-CO': 200
    }
    preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
    fdm.sample_wise_preprocessing(
        flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
        )

# Define channels that were used for model training
channels = trainchannels

# Extract the processed data matrices from the AnnData objects
data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
# Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
num_events = [x.shape[0] for x in data_matrices]
starting_indices = np.cumsum([0, ] + num_events)
# Concatenate
x_test = np.concatenate(data_matrices, axis=0)

# Build dataframe `df_calc` from the concatenated 'no_trafo' layer (for populaton statistics)
# Collect per-sample non-transformed matrices for all variables (use var names)
var_names = list(fdm.anndata_list_[0].var_names)
no_trafo_matrices = [adata.layers['no_trafo'] for adata in fdm.anndata_list_]
# Concatenate into one matrix
no_trafo_concat = np.concatenate(no_trafo_matrices, axis=0)
# Create sample labels for each event using starting indices and inference file names
sample_labels = []
for i, n in enumerate(num_events):
    sample_labels.extend([inference_files[i]] * n)
# Build DataFrame using all variable names
df_calc = pd.DataFrame(no_trafo_concat, columns=var_names)
df_calc['sample'] = sample_labels

time_b = datetime.now()
timeload = time_b - timestart

def load_model_by_prefix(model_class, prefix, model_dir):
    matching_files = sorted(
        fn for fn in os.listdir(model_dir)
        if fn.lower().endswith('.pkl') and fn.lower().startswith(prefix.lower())
    )
    if not matching_files:
        raise FileNotFoundError(f"No model files with prefix '{prefix}' found in {model_dir}")

    selected_file = matching_files[0]
    print(f"Loading model from {selected_file}")
    return model_class.load(filename=selected_file, filepath=model_dir), selected_file

# Load the previously trained models
som_clf, som_model_file = load_model_by_prefix(SOMClassifier, 'som_classifier', './data/models')
mlp_clf, mlp_model_file = load_model_by_prefix(MLPClassifier, 'mlp_classifier', './data/models')

# Make prediction for the test data
print('make predictions for test data...')
y_pred_som = som_clf.predict(x_test)
y_pred_mlp = mlp_clf.predict(x_test)

# Keep a copy of the full concatenated MLP predictions to attach to df_calc
y_pred_mlp_full = np.array(y_pred_mlp)

# Change predictions back into sample-wise format (input format required by export function)
y_pred_som = [y_pred_som[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]
y_pred_mlp = [y_pred_mlp[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]

add_columns = [y_pred_som, y_pred_mlp]
add_columns_names = ['pred_som', 'pred_mlp']

time_c = datetime.now()
timepredict = time_c - time_b

# Attach concatenated MLP predictions to df_calc
try:
    df_calc['y_pred_mlp'] = y_pred_mlp_full
    print('updated df_calc with y_pred_mlp internally')
except Exception as e:
    print(f'could not attach y_pred_mlp to df_calc: {e}')

# Generate per-sample summary dataframe `df_calc_results` with counts per mlp class
try:
    # Ensure predictions are integers
    df_calc['y_pred_mlp'] = df_calc['y_pred_mlp'].astype(int)
    counts = df_calc.groupby('sample')['y_pred_mlp'].value_counts().unstack(fill_value=0)
    # Rename columns to mlp_{label}
    counts.columns = [f'mlp_{int(c)}' for c in counts.columns]
    df_calc_results = counts.reset_index()
    
    # Also compute median values for selected `calcchannels` per sample and mlp population
    try:
        medians = df_calc.groupby(['sample', 'y_pred_mlp'])[calcchannels].median()
        if calc_populations:
            try:
                selected_populations = [int(p) for p in calc_populations]
                medians = medians.loc[medians.index.get_level_values('y_pred_mlp').isin(selected_populations)]
            except Exception:
                print(f'warning: calc_populations must contain numeric population labels. Ignoring selection and computing all populations.')
        medians_reset = medians.reset_index()
        # Build mapping of sample -> {colname: value}
        median_map = {}
        median_cols = set()
        for _, r in medians_reset.iterrows():
            sample = r['sample']
            label = int(r['y_pred_mlp'])
            for ch in calcchannels:
                colname = f'mlp_{label}_{ch}_median'
                median_cols.add(colname)
                median_map.setdefault(sample, {})[colname] = r[ch]

        # Add median columns to df_calc_results (one row per sample)
        for colname in sorted(median_cols):
            df_calc_results[colname] = df_calc_results['sample'].map(lambda s: median_map.get(s, {}).get(colname, np.nan))

        # Reorder median columns by channel first, then population
        population_labels = sorted({int(label) for _, label in medians.index})
        channel_order = sorted(calcchannels)
        ordered_median_cols = []
        for ch in channel_order:
            for label in population_labels:
                colname = f'mlp_{label}_{ch}_median'
                if colname in df_calc_results.columns:
                    ordered_median_cols.append(colname)

        base_columns = [c for c in df_calc_results.columns if c not in set(ordered_median_cols)]
        df_calc_results = df_calc_results[base_columns + ordered_median_cols]
    except Exception as e:
        print(f'could not compute per-population medians: {e}')
    summary_outfile = os.path.join(save_path, f'df_calc_results_{date_time_str}.csv')
    df_calc_results.to_csv(summary_outfile, sep=';', decimal=',', index=False)
    print(f'saved df_calc_results to {summary_outfile}')
except Exception as e:
    print(f'could not create df_calc_results: {e}')

# If dimensionality reductions should be computed as well, set: compute_dim_red = True
if compute_dim_red:

    # --- SOM
    print('compute SOM...')
    _, x_som, _, _ = som_clf.transform(x_test)

    # --- UMAP
    # print('compute UMAP...')
    # umap_model = UMAP(n_components=2, n_jobs=-1)
    # x_umap = umap_model.fit_transform(x_test)

    # --- t-SNE
    print('compute t-SNE...')
    tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
    x_tsne=x_tsne = tsne_model.fit(x_test)
    # if remapping to previous TSNE, use instead
    # tsne_old = pickle.load(open('./results/workflow_step_wise_supervised_training/tsne_embedding.pkl', 'rb'))
    # x_tsne = tsne_old.transform(x_test)

    time_d = datetime.now()
    timetsne = time_d - time_c

    # Change back into sample-wise format (input format required by export function)
    x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
    # x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    # x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
    x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

    add_columns += [
        x_soms_1, x_soms_2,
        x_tsnes_1, x_tsnes_2
    ]
    add_columns_names += ['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2']

# Export to FCS
export_to_fcs(
    data_list=fdm.anndata_list_,  # Export the test samples
    layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
    sample_wise=False,  # Export one FCS in which the test samples are concatenated
    add_columns=add_columns,  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
    add_columns_names=add_columns_names,  # Add names for added columns
    scale_columns=add_columns_names,  # Select added columns for scaling (all that were added to the file)
    val_range=val_range,  # Range to which selected columns are scaled to
    save_path=save_path,
    save_filenames=f'inference_data_{date_time_str}.fcs'
)

timetotal = datetime.now()-timestart
with open(os.path.join(save_path, f'fcs_inference_{date_time_str}.txt'), 'a') as f:
    f.write(f'"inference_files" {date_time_str}: \n')
    for items in inference_files:
        f.write(items + "\n")
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"som classifier file": {som_model_file}\n')
    f.write(f'"mlp classifier file": {mlp_model_file}\n')
    f.write(f'"val_range": {val_range}\n')
    f.write(f'"trafo_arcsinh": {trafo_arcsinh} "arcsinh cofactor": {arcsinh_div}\n')
    f.write (f'"dimreduction and fcs output generated": {compute_dim_red}\n')
    f.write(f'"population statistics provided in file df_calc_results_{date_time_str}.csv"\n')
    f.write(f'"events per population for all populations and samples": \n')
    f.write(f'"channel medians per population for populations {calc_populations} and channels {calcchannels}": \n')
    f.write(f'"time data load": {timeload}\n')
    f.write(f'"time prediction": {timepredict}\n')
    # f.write(f'"time t-SNE": {timetsne}\n')
    f.write(f'"timetotal": {timetotal}\n')