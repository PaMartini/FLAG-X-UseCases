# uses inference from supervised training to preselect populations of interest from new data
# populations of less interest will be downsampled to obtain smaller files for further analysis
# maximum population sizes are defined in config_xxx.yml
# one fcs output file with compensated raw data is exported per input file, with downsampling applied
# for mere fcs export, set compute_dim_red = False
# runs with fcs or csv (english format) input files, several files at a time 
# check that preprocessing (trafo)and channels are identical to training samples
# tested and running 2026-08-09

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
from openTSNE import TSNE
import anndata as ad

# --- selected Parameters from YAML files, select and configure suitable file, parameters identical to training!-------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Bcell.yml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
save_path = config.get('save_path_preselect')
data_path = config.get('path_preselect')
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
compute_dim_red = config.get('compute_dim_red') 
size_pop_1 = config.get('size_pop_1')  # number of events from population 1 to keep 
size_pop_2 = config.get('size_pop_2')  # number of events from population 2 to keep
size_pop_3 = config.get('size_pop_3')  # number of events from population 3 to keep
include_predict_columns = config.get('include_predict_columns')  # Whether to include prediction columns in the exported FCS file

# --- Define path where results are saved to
save_path = save_path
os.makedirs(save_path, exist_ok=True)

# Define path to inference data
inference_data_path = data_path

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

# --- Apply spillover compensation if the FCS files contain a spillover matrix
print('applying spillover compensation...')
try:
    fdm.sample_wise_compensation()
    if fdm.compensation_log_ is not None:
        print(fdm.compensation_log_.to_string(index=False))
except Exception as e:
    print(f'compensation step failed for one or more files: {e}')

# --- Apply preprocessing transformation to each sample, same as for training data
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

# Process each input file independently and export one preselected FCS per input
print('make predictions for each input file...')
time_c = datetime.now()
timepredict = time_c - time_b

population_limits = {
    1: int(size_pop_1),
    2: int(size_pop_2),
    3: int(size_pop_3),
}
rng = np.random.default_rng(42)

export_data_list = []
pred_som_cols = []
pred_mlp_cols = []
summary_rows = []

if compute_dim_red:
    som_1_cols = []
    som_2_cols = []
    tsne_1_cols = []
    tsne_2_cols = []

for i, adata in enumerate(fdm.anndata_list_):
    sample_name = inference_files[i]
    x_sample = adata[:, channels].X

    y_pred_som_sample = som_clf.predict(x_sample)
    y_pred_mlp_sample = mlp_clf.predict(x_sample)

    kept_indices = []
    population_counts = {}
    population_downsample_factors = {}

    for label in np.unique(y_pred_mlp_sample):
        label_int = int(label)
        label_indices = np.flatnonzero(y_pred_mlp_sample == label_int)
        count = int(label_indices.size)
        population_counts[label_int] = count

        max_events = population_limits.get(label_int)
        if max_events is None or count <= max_events:
            selected_indices = label_indices
            factor = 1.0
        else:
            selected_indices = rng.choice(label_indices, size=max_events, replace=False)
            factor = count / max_events

        population_downsample_factors[label_int] = float(factor)
        kept_indices.extend(selected_indices.tolist())

    if not kept_indices:
        kept_indices = np.arange(len(y_pred_mlp_sample), dtype=int)
    kept_indices = np.sort(np.asarray(kept_indices, dtype=int))

    adata_subset = adata[kept_indices, :].copy()
    export_data_list.append(adata_subset)

    pred_som_cols.append(y_pred_som_sample[kept_indices])
    pred_mlp_cols.append(y_pred_mlp_sample[kept_indices])

    if compute_dim_red:
        x_sample_subset = x_sample[kept_indices]
        _, x_som_sample, _, _ = som_clf.transform(x_sample_subset)
        tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
        x_tsne_sample = tsne_model.fit(x_sample_subset)

        som_1_cols.append(x_som_sample[:, 0])
        som_2_cols.append(x_som_sample[:, 1])
        tsne_1_cols.append(x_tsne_sample[:, 0])
        tsne_2_cols.append(x_tsne_sample[:, 1])

    summary_row = {'sample': sample_name}
    for label in sorted(population_counts):
        summary_row[f'mlp_{label}'] = population_counts[label]
        summary_row[f'mlp_{label}_downsample_factor'] = population_downsample_factors[label]
    summary_rows.append(summary_row)

# Create df_calc_results with population counts and downsampling fractions
try:
    df_calc_results = pd.DataFrame(summary_rows)
    summary_outfile = os.path.join(save_path, f'fcs_pop_preselect_results_{date_time_str}.csv')
    df_calc_results.to_csv(summary_outfile, sep=';', decimal=',', index=False)
    print(f'saved df_calc_results to {summary_outfile}')
except Exception as e:
    print(f'could not create df_calc_results: {e}')

# Build export columns only if requested
add_columns = []
add_columns_names = []
if include_predict_columns:
    add_columns.extend([pred_som_cols, pred_mlp_cols])
    add_columns_names.extend(['pred_som', 'pred_mlp'])

if compute_dim_red:
    add_columns.extend([som_1_cols, som_2_cols, tsne_1_cols, tsne_2_cols])
    add_columns_names.extend(['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2'])

# Export one FCS per input file with the preselection and downsampling applied (compensated raw data)
save_filenames = [f'preselect_{os.path.splitext(fn)[0]}.fcs' for fn in inference_files]
export_to_fcs(
    data_list=export_data_list,
    layer_key='no_trafo',
    sample_wise=True,
    add_columns=add_columns if add_columns else None,
    add_columns_names=add_columns_names if add_columns_names else None,
    scale_columns=add_columns_names if add_columns_names else None,
    val_range=val_range,
    save_path=save_path,
    save_filenames=save_filenames,
)

timetotal = datetime.now()-timestart
with open(os.path.join(save_path, f'fcs_pop_preselect_{date_time_str}.txt'), 'a') as f:
    f.write(f'"files for preselction of cell populations" {date_time_str}: \n')
    for items in inference_files:
        f.write(items + "\n")
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"som classifier file": {som_model_file}\n')
    f.write(f'"mlp classifier file": {mlp_model_file}\n')
    f.write(f'"val_range": {val_range}\n')
    f.write(f'"trafo_arcsinh": {trafo_arcsinh} "arcsinh cofactor": {arcsinh_div}\n')
    f.write(f'"channel cutoff for log trafo": {channel_name_to_cutoff}\n')
    f.write(f'"lin_trafo_FSSS": {lin_trafo_FSSS}\n')
    f.write (f'"dimreduction generated": {compute_dim_red}\n')
    f.write(f'"population statistics and downsampling factors provided in file df_calc_results_{date_time_str}.csv"\n')
    f.write(f'"one downsampled FCS output is exported per input file (compensated data)" \n')
    f.write(f'"time data load": {timeload}\n')
    # f.write(f'"time prediction": {timepredict}\n')
    # f.write(f'"time t-SNE": {timetsne}\n')
    f.write(f'"timetotal": {timetotal}\n')
