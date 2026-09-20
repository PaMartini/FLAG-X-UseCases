# uses inference from supervised training to preselect populations of interest from new fcs data
# populations of less interest will be downsampled to obtain smaller files for further analysis
# maximum input size per sample and population sizes are defined in config_xxx.yml
# one fcs output file with compensated raw data is exported per input file, with downsampling applied
# for mere fcs export, set compute_dim_red = False
# runs with fcs or csv (english format) input files, several files at a time 
# check that preprocessing (trafo)and channels are identical to training
# modified due to need of memory optimization 2026-09-18

print('loading scripts and data...')
import os
import gc
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

# --- select YAML file! ---
# configure suitable file, parameters identical to training!
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Tcell.yml')
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
size_per_sample = config.get('size_per_sample')  # Maximum number of events per sample to be used 
trafo_arcsinh = config.get('trafo_arcsinh')
arcsinh_div = config.get('arcsinh_div') # only applied if trafo_arcsinh is True
channel_name_to_cutoff = config.get('channel_name_to_cutoff') # only applied if trafo_arcsinh is False
multiply_FSSS = config.get('multiply_FSSS')
upscale_val_list = config.get('upscale_val') # only applied if multiply_FSSS is True
upscale_val = tuple(upscale_val_list)
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

# Define channels that were used for model training
channels = trainchannels

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

summary_rows = []

# Process each input file completely on its own: load -> preprocess -> predict -> downsample -> export.
# This way only one file's data is held in memory at a time instead of loading all files upfront.
for i, fn in enumerate(inference_files):
    sample_name = fn
    print(f'--- processing file {i + 1}/{len(inference_files)}: {sample_name} ---')

    # --- Load a single file into memory
    fdm = FlowDataManager(
        data_file_names=[fn],
        data_file_type=None,  # Is inferred from the filename ending
        data_file_path=inference_data_path,
        verbosity=1
    )
    fdm.load_data_files_to_anndata()

    # --- Apply spillover compensation if the FCS file contains a spillover matrix
    print('applying spillover compensation...')
    try:
        fdm.sample_wise_compensation()
        if fdm.compensation_log_ is not None:
            print(fdm.compensation_log_.to_string(index=False))
    except Exception as e:
        print(f'compensation step failed for file {sample_name}: {e}')

    # sample_wise_compensation() stores a full-size copy of the pre-compensation data in an
    # 'uncompensated' layer that this script never uses. Drop it right away to avoid holding
    # a third full-size copy of the data (X + 'no_trafo' + 'uncompensated') in memory at once.
    for _adata in fdm.anndata_list_:
        _adata.layers.pop('uncompensated', None)

    # --- Apply preprocessing transformation, same as for training data
    # store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
    if trafo_arcsinh:
        preprocessing_kwargs = {'cofactor': arcsinh_div}
        fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
    else:
        preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
        fdm.sample_wise_preprocessing(
            flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
            )

    adata = fdm.anndata_list_[0]

    # Optional: 'FS INT' and 'SS INT' will be scaled up (makes only sense if the trafo is set to log, not arcsinh)
    if multiply_FSSS:
        if 'FS INT' in adata.var_names:
            adata[:, 'FS INT'].X = (adata[:, 'FS INT'].X - upscale_val['FSsub']) * upscale_val['FSmult']
        if 'SS INT' in adata.var_names:
            adata[:, 'SS INT'].X = (adata[:, 'SS INT'].X - upscale_val['SSsub']) * upscale_val['SSmult']

    # --- Downsample each sample to a target number of events
    # sample_wise_downsampling() replaces fdm.anndata_list_[0] with a new (downsampled) AnnData
    # object rather than mutating the existing one in place, so `adata` must be re-fetched here 
    fdm.sample_wise_downsampling(data_set='all', target_num_events=size_per_sample)
    adata = fdm.anndata_list_[0]

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

    pred_som_col = y_pred_som_sample[kept_indices]
    pred_mlp_col = y_pred_mlp_sample[kept_indices]

    x_sample_subset = x_sample[kept_indices] if compute_dim_red else None

    # adata and x_sample hold the full (non-downsampled) event matrix and are no longer needed
    # once the downsampled subset/predictions have been extracted; free them before running the
    # comparatively heavy SOM/t-SNE transforms and export below. fdm.anndata_list_ still holds a
    # reference to the same object, so it must be cleared too, or the memory is not actually freed.
    fdm.anndata_list_[0] = None
    del adata, x_sample, y_pred_som_sample, y_pred_mlp_sample
    gc.collect()

    add_columns = []
    add_columns_names = []
    if include_predict_columns:
        add_columns.extend([[pred_som_col], [pred_mlp_col]])
        add_columns_names.extend(['pred_som', 'pred_mlp'])

    if compute_dim_red:
        _, x_som_sample, _, _ = som_clf.transform(x_sample_subset)
        tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
        x_tsne_sample = tsne_model.fit(x_sample_subset)

        add_columns.extend([
            [x_som_sample[:, 0]], [x_som_sample[:, 1]], [x_tsne_sample[:, 0]], [x_tsne_sample[:, 1]]
        ])
        add_columns_names.extend(['SOM_1', 'SOM_2', 'TSNE_1', 'TSNE_2'])

    # --- Export this single file's preselection and downsampling result immediately (compensated raw data)
    export_to_fcs(
        data_list=[adata_subset],
        layer_key='no_trafo',
        sample_wise=True,
        add_columns=add_columns if add_columns else None,
        add_columns_names=add_columns_names if add_columns_names else None,
        scale_columns=add_columns_names if add_columns_names else None,
        val_range=val_range,
        save_path=save_path,
        save_filenames=[f'preselect_{os.path.splitext(fn)[0]}.fcs'],
    )

    summary_row = {'sample': sample_name}
    for label in sorted(population_counts):
        summary_row[f'mlp_{label}'] = population_counts[label]
        summary_row[f'mlp_{label}_downsample_factor'] = population_downsample_factors[label]
    summary_rows.append(summary_row)

    # Release the completed file before loading the next one.
    # (adata/x_sample/y_pred_*_sample were already freed above, right after they became unneeded.)
    del (
        fdm, adata_subset, x_sample_subset,
        pred_som_col, pred_mlp_col,
        kept_indices, population_counts, population_downsample_factors,
        add_columns, add_columns_names,
    )
    if compute_dim_red:
        del x_som_sample, x_tsne_sample, tsne_model
    gc.collect()

# Create df_calc_results with population counts and downsampling fractions
try:
    df_calc_results = pd.DataFrame(summary_rows)
    summary_outfile = os.path.join(save_path, f'fcs_pop_preselect_results_{date_time_str}.csv')
    df_calc_results.to_csv(summary_outfile, sep=';', decimal=',', index=False)
    print(f'saved df_calc_results to {summary_outfile}')
except Exception as e:
    print(f'could not create df_calc_results: {e}')

timetotal = datetime.now()-timestart
with open(os.path.join(save_path, f'fcs_pop_preselect_{date_time_str}.txt'), 'a') as f:
    f.write(f'"files for preselction of cell populations" {date_time_str}: \n')
    for items in inference_files:
        f.write(items + "\n")
    f.write (f' "large files downsampled to" {size_per_sample} "events"\n')
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"som classifier file": {som_model_file}\n')
    f.write(f'"mlp classifier file": {mlp_model_file}\n')
    f.write(f'"val_range": {val_range}\n')
    f.write(f'"trafo_arcsinh": {trafo_arcsinh} "arcsinh cofactor": {arcsinh_div}\n')
    f.write(f'"channel cutoff for log trafo": {channel_name_to_cutoff}\n')
    f.write(f'"multiply_FSSS": {multiply_FSSS}\n')
    f.write (f'"dimreduction generated": {compute_dim_red}\n')
    f.write(f'"population statistics and downsampling factors provided in file df_calc_results_{date_time_str}.csv"\n')
    f.write(f'"one downsampled FCS output is exported per input file, processed and saved one file at a time (compensated data)" \n')
    f.write(f'"time data load": {timeload}\n')
    # f.write(f'"time prediction": {timepredict}\n')
    # f.write(f'"time t-SNE": {timetsne}\n')
    f.write(f'"timetotal": {timetotal}\n')

