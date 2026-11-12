# inference script for flow cytometry data with trained SOM and MLP classifiers
# calculates population statistics.
# populations to analyze can be selected in config_Sysmex.yml 
# exports fcs with population predictions and optional SOM and TSNE display. 
# for mere caluclation of population statistics, set calcTSNE = False
# consider downsampling, if only statistics are of interest
# runs with fcs, lmd or csv (english format) input files, not fcs and csv mixed 
# check that preprocessing (trafo)and channels are identical to training samples
# batch-wise processing of large number of files implemented, with user-defined maximum number of rows per batch.
# downsampling enabled, tested and running 2026-11-1 (including lmd)

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
from channel_harmonize_module import prepare_channels
from import_lmd_stubb import add_pns_to_fcs30
from flagx.gating import SOMClassifier, MLPClassifier
from flagx.dimred import UMAP
from openTSNE import TSNE
import anndata as ad

# --- selected Parameters from YAML files, select and configure suitable file, parameters identical to training!-------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Imst_rout.yml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
inference_data_path = config.get('path_inference')
path_models = config.get('path_models')
save_path_inference = config.get('save_path_inference')
delete_channels = config.get('delete_channels')
harmonize_channels = config.get('harmonize_channels')
all_channels = config.get('all_channels')
trainchannels = config.get('trainchannels')
size_per_sample = config.get('size_per_sample')  # Maximum number of events per input file
max_rows_per_batch = config.get('max_rows_per_batch')  # Maximum number of rows to read in at once for inference
SOM_dim = tuple(config.get('SOM_dim'))  # Dimensions of the SOM grid. 10x10 for fast testing, 25x25 to 30x30 for better resolution
SOM_epochs = config.get('SOM_epochs')  # Number of epochs for SOM training. default 100 for smaller grids, up to 1000
val_range_list = config.get('val_range') 
val_range = tuple(val_range_list)  
trafo_arcsinh = config.get('trafo_arcsinh')
arcsinh_div = config.get('arcsinh_div')
channel_name_to_cutoff = config.get('channel_name_to_cutoff')
multiply_FSSS = config.get('multiply_FSSS')
upscale_val_list = config.get('upscale_val') # only applied if multiply_FSSS is True
upscale_val = dict(upscale_val_list)
compute_TSNE = config.get('calcTSNE')
calcchannels = config.get('calcchannels') # List of channels for which to calculate median intensities per population
calc_populations = config.get('calc_populations')  # Optional list of predicted populations for which to compute channel medians
compute_SOM = config.get('show_SOM')  

# --- Define path where results are saved to
save_path = save_path_inference
os.makedirs(save_path, exist_ok=True)

# Define path to inference data
inference_data_path = inference_data_path

# Get list of flow cytometry files in the data directory (.fcs, .lmd or .csv; do not mix file types)
inference_files = sorted([
    fn for fn in os.listdir(inference_data_path)
    if fn.lower().endswith(('.fcs', '.lmd', '.csv'))
])
if not inference_files:
    raise FileNotFoundError(f'No inference files found in {inference_data_path}')

is_lmd = all(fn.lower().endswith('.lmd') for fn in inference_files)
if not is_lmd and any(fn.lower().endswith('.lmd') for fn in inference_files):
    raise ValueError('LMD files cannot be mixed with other file types in the inference directory')

# --- Data loading and batching during actual FCS processing
MAX_ROWS_PER_BATCH = max_rows_per_batch
results_outfile_joint = os.path.join(save_path, f'df_calc_results_{date_time_str}.csv')

# Initialize timing variables
timeload = None
timepredict = None
original_channels_by_file = {}
original_rows_by_file = {}


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


som_clf, som_model_file = load_model_by_prefix(SOMClassifier, 'som_classifier', path_models)
mlp_clf, mlp_model_file = load_model_by_prefix(MLPClassifier, 'mlp_classifier', path_models)

time_b = datetime.now()
timeload = time_b - timestart


def process_batch(fdm, batch_number):
    batch_file_list = [adata.uns['filename'] for adata in fdm.anndata_list_]
    print(f'\n{"="*70}')
    print(f'BATCH {batch_number}: Processing {len(batch_file_list)} file(s)')
    print(f'{"="*70}\n')

    print('applying spillover compensation...')
    try:
        fdm.sample_wise_compensation()
        if fdm.compensation_log_ is not None:
            print(fdm.compensation_log_.to_string(index=False))
    except Exception as e:
        print(f'compensation step failed for one or more files: {e}')

    original_channels_by_file.update({
        adata.uns['filename']: list(adata.var_names) for adata in fdm.anndata_list_
    })
    fdm.anndata_list_ = prepare_channels(
        fdm.anndata_list_, delete_channels, harmonize_channels, all_channels
    )

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
            adata.X[:, adata.var_names.get_loc('FS INT')] = (adata[:, 'FS INT'].X.ravel() - upscale_val['FSsub']) * upscale_val['FSmult']
        if 'SS INT' in adata.var_names:
            adata.X[:, adata.var_names.get_loc('SS INT')] = (adata[:, 'SS INT'].X.ravel() - upscale_val['SSsub']) * upscale_val['SSmult']

    channels = trainchannels
    data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
    num_events = [x.shape[0] for x in data_matrices]
    starting_indices = np.cumsum([0, ] + num_events)
    x_test = np.concatenate(data_matrices, axis=0)

    var_names = list(fdm.anndata_list_[0].var_names)
    no_trafo_matrices = [adata.layers['no_trafo'] for adata in fdm.anndata_list_]
    no_trafo_concat = np.concatenate(no_trafo_matrices, axis=0)

    sample_labels = []
    for i, n in enumerate(num_events):
        sample_labels.extend([batch_file_list[i]] * n)

    df_calc = pd.DataFrame(no_trafo_concat, columns=var_names)
    df_calc['sample'] = sample_labels
    df_calc['batch_number'] = batch_number

    print('make predictions for test data...')
    y_pred_som = som_clf.predict(x_test)
    y_pred_mlp = mlp_clf.predict(x_test)
    y_pred_mlp_full = np.array(y_pred_mlp)

    y_pred_som = [y_pred_som[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]
    y_pred_mlp = [y_pred_mlp[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]

    add_columns = [y_pred_som, y_pred_mlp]
    add_columns_names = ['pred_som', 'pred_mlp']

    try:
        df_calc['y_pred_mlp'] = y_pred_mlp_full
        print('updated df_calc with y_pred_mlp internally')
    except Exception as e:
        print(f'could not attach y_pred_mlp to df_calc: {e}')

    try:
        df_calc['y_pred_mlp'] = df_calc['y_pred_mlp'].astype(int)
        counts = df_calc.groupby(['sample', 'batch_number'])['y_pred_mlp'].value_counts().unstack(fill_value=0)
        counts.columns = [f'mlp_{int(c)}' for c in counts.columns]
        df_calc_results_batch = counts.reset_index()

        try:
            medians = df_calc.groupby(['sample', 'batch_number', 'y_pred_mlp'])[calcchannels].median()
            if calc_populations:
                try:
                    selected_populations = [int(p) for p in calc_populations]
                    medians = medians.loc[medians.index.get_level_values('y_pred_mlp').isin(selected_populations)]
                except Exception:
                    print(f'warning: calc_populations must contain numeric population labels. Ignoring selection and computing all populations.')
            medians_reset = medians.reset_index()
            median_map = {}
            median_cols = set()
            for _, r in medians_reset.iterrows():
                sample = r['sample']
                batch_id = int(r['batch_number'])
                label = int(r['y_pred_mlp'])
                for ch in calcchannels:
                    colname = f'mlp_{label}_{ch}_median'
                    median_cols.add(colname)
                    median_map.setdefault((sample, batch_id), {})[colname] = r[ch]

            for colname in sorted(median_cols):
                df_calc_results_batch[colname] = df_calc_results_batch.apply(
                    lambda row: median_map.get((row['sample'], int(row['batch_number'])), {}).get(colname, np.nan),
                    axis=1,
                )

            population_labels = sorted({int(label) for _, _, label in medians.index})
            channel_order = sorted(calcchannels)
            ordered_median_cols = []
            for ch in channel_order:
                for label in population_labels:
                    colname = f'mlp_{label}_{ch}_median'
                    if colname in df_calc_results_batch.columns:
                        ordered_median_cols.append(colname)

            base_columns = [c for c in df_calc_results_batch.columns if c not in set(ordered_median_cols)]
            df_calc_results_batch = df_calc_results_batch[base_columns + ordered_median_cols]
        except Exception as e:
            print(f'could not compute per-population medians: {e}')

        if batch_number == 1:
            df_calc_results_batch.to_csv(results_outfile_joint, sep=';', decimal=',', index=False)
            print(f'created df_calc_results file with batch {batch_number}')
        else:
            df_calc_results_batch.to_csv(results_outfile_joint, sep=';', decimal=',', index=False, mode='a', header=False)
            print(f'appended batch {batch_number} results to {results_outfile_joint}')
    except Exception as e:
        print(f'could not create batch {batch_number} df_calc_results: {e}')

    if compute_SOM:
        print('compute SOM...')
        _, x_som, _, _ = som_clf.transform(x_test)

        x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
        x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
        
        add_columns += [x_soms_1, x_soms_2]
        add_columns_names += ['SOM_1', 'SOM_2']

    if compute_TSNE:
        print('compute t-SNE...')
        tsne_model = TSNE(n_components=2, n_jobs=-1, verbose=True)
        x_tsne = tsne_model.fit(x_test)

        x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
        x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

        add_columns += [x_tsnes_1, x_tsnes_2]
        add_columns_names += ['TSNE_1', 'TSNE_2']

    export_to_fcs(
        data_list=fdm.anndata_list_,
        layer_key='no_trafo',
        sample_wise=False,
        add_columns=add_columns,
        add_columns_names=add_columns_names,
        scale_columns=add_columns_names,
        val_range=val_range,
        save_path=save_path,
        save_filenames=f'inference_data_batch{batch_number}_{date_time_str}.fcs'
    )

    print(f'Batch {batch_number} processed and exported.')

    print(f'Cleaning memory for batch {batch_number}...')
    del fdm, x_test, y_pred_som, y_pred_mlp, add_columns, add_columns_names
    if compute_SOM:
        del x_som, x_soms_1, x_soms_2
    del df_calc
    import gc
    gc.collect()


# Load and downsample each file once, before batching and subsequent calculations.
# Rule: when a file would exceed the limit, that file is still processed in the current batch,
# and the following file starts a fresh batch.
batch_number = 0
batch_fdm = None
batch_rows = 0
input_type = os.path.splitext(inference_files[0])[1][1:].lower()

for fn in inference_files:
    if is_lmd:
        file_fdm = add_pns_to_fcs30(data_path=inference_data_path, lmd_fns=[fn], verbosity=1)
    else:
        file_fdm = FlowDataManager(
            data_file_names=[fn],
            data_file_type=input_type,
            data_file_path=inference_data_path,
            verbosity=1,
        )
        file_fdm.load_data_files_to_anndata()
    original_rows_by_file[fn] = sum(adata.n_obs for adata in file_fdm.anndata_list_)
    if not file_fdm.anndata_list_:
        continue
    file_fdm.sample_wise_downsampling(data_set='all', target_num_events=size_per_sample)
    batch_rows += sum(adata.n_obs for adata in file_fdm.anndata_list_)
    if batch_fdm is None:
        batch_fdm = file_fdm
    else:
        batch_fdm.anndata_list_.extend(file_fdm.anndata_list_)
    del file_fdm

    if len(batch_fdm.anndata_list_) > 1 and batch_rows > MAX_ROWS_PER_BATCH:
        batch_number += 1
        print(f'Batch limit reached ({MAX_ROWS_PER_BATCH} rows); processing the current file with this batch and starting a fresh batch for the next file.')
        process_batch(batch_fdm, batch_number)
        batch_fdm = None
        batch_rows = 0

# Process remaining files after the loop
if batch_fdm is not None:
    batch_number += 1
    process_batch(batch_fdm, batch_number)

print(f'\n{"="*70}')
print(f'All batches processed. Population statistics saved to:')
print(f'{results_outfile_joint}')
print(f'{"="*70}\n')

timetotal = datetime.now()-timestart

with open(os.path.join(save_path, f'fcs_inference_{date_time_str}.txt'), 'a') as f:
    f.write(f'"inference_files and size" {date_time_str}: \n')
    for items in inference_files:
        f.write(f'{items} ({original_rows_by_file.get(items, 0)} events)\n')
        if harmonize_channels and items in original_channels_by_file:
            f.write(f'"original channels": {original_channels_by_file[items]}\n')
    original_channel_names = {
        channel for channels in original_channels_by_file.values() for channel in channels
    }
    deleted_columns = list(dict.fromkeys(
        name for name in (delete_channels or []) if name in original_channel_names
    ))
    f.write(f'"deleted_channels": {deleted_columns if deleted_columns else "none"}\n')
    if harmonize_channels:
        f.write(f'"harmonized channels": {all_channels}\n')
    f.write(f'"training channels": {trainchannels}\n')
    f.write(f'"som classifier file": {som_model_file}\n')
    f.write(f'"mlp classifier file": {mlp_model_file}\n')
    f.write(f'"processed in" {batch_number} "batches": \n')
    f.write(f'"large samples downsampled to": {size_per_sample}\n')
    f.write(f'"val_range": {val_range}\n')
    f.write(f'"trafo_arcsinh": {trafo_arcsinh}\n')
    if trafo_arcsinh:
        f.write(f'"arcsinh cofactor": {arcsinh_div}\n')
    else:
        f.write(f'"channel cutoff for log trafo": {channel_name_to_cutoff}\n')
        f.write(f'"scale up FSSS": {multiply_FSSS}\n')
    if multiply_FSSS:
        f.write(f'"FSSS upscale factor": {upscale_val_list}\n')
    f.write (f'"dimreduction for fcs output generated-SOM": {compute_SOM} "TSNE": {compute_TSNE}\n')
    f.write(f'"population statistics provided in file df_calc_results_{date_time_str}.csv"\n')
    f.write(f'"events per population (as defined by MLP) for all populations and samples": \n')
    f.write(f'"channel medians per population for populations {calc_populations} and channels {calcchannels}": \n')
    f.write(f'"time data load": {timeload}\n')
    f.write(f'"timetotal": {timetotal}\n')
