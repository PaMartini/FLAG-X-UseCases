# workflow_step_wise_supervised_inference
# calculates population statistics and exports fcs with population predictions and optional dim reduction. 
# populations to analyze can be selected in config_Sysmex.yml
# for mere caluclation of population statistics, set compute_dim_red = False
# runs with fcs or csv (english format) input files, several files at a time 
# check that preprocessing (trafo)and channels are identical to training samples
# TSNE inference to trained TSNE model included as an option, working, but not very well.
# batch-weise processing of large datasets implemented, with user-defined maximum number of rows per batch.
# tested and running 2026-08-18

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
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Bcell.yml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
trainchannels = config.get('trainchannels')
inference_data_path = config.get('path_inference')
save_path_inference = config.get('save_path_inference')
max_rows_per_batch = config.get('max_rows_per_batch')  # Maximum number of rows to read in at once for inference
SOM_dim = tuple(config.get('SOM_dim'))  # Dimensions of the SOM grid. 10x10 for fast testing, 25x25 to 30x30 for better resolution
SOM_epochs = config.get('SOM_epochs')  # Number of epochs for SOM training. default 100 for smaller grids, up to 1000
val_range_list = config.get('val_range') 
val_range = tuple(val_range_list)  
trafo_arcsinh = config.get('trafo_arcsinh')
arcsinh_div = config.get('arcsinh_div')
channel_name_to_cutoff = config.get('channel_name_to_cutoff')
lin_trafo_FSSS = config.get('lin_trafo_FSSS')
calcTSNE = config.get('calcTSNE')
calcchannels = config.get('calcchannels') # List of channels for which to calculate median intensities per population
calc_populations = config.get('calc_populations')  # Optional list of predicted populations for which to compute channel medians
compute_dim_red = config.get('compute_dim_red')  

# --- Define path where results are saved to
save_path = save_path_inference
os.makedirs(save_path, exist_ok=True)

# Define path to inference data
inference_data_path = inference_data_path

# Get list of flow cytometry files in the data directory (prefer .fcs files, fall back to .csv for compatibility)
inference_files = sorted([
    fn for fn in os.listdir(inference_data_path)
    if fn.lower().endswith(('.fcs', '.csv'))
])
if not inference_files:
    raise FileNotFoundError(f'No inference files found in {inference_data_path}')

# --- Data loading and processing with batch limiting 
MAX_ROWS_PER_BATCH = max_rows_per_batch
batch_number = 0
remaining_files = list(inference_files)

def estimate_file_rows(fname, data_path):
    """Estimate number of rows in a data file"""
    file_path = os.path.join(data_path, fname)
    try:
        if fname.lower().endswith('.fcs'):
            # Try to read FCS file to get row count
            temp_adata = ad.read_h5ad(file_path) if file_path.endswith('.h5ad') else None
            if temp_adata is None:
                # Fallback: assume reasonable size for FCS
                return 100000
            return temp_adata.n_obs
        else:  # CSV
            return len(pd.read_csv(file_path))
    except Exception as e:
        print(f"  Warning: Could not estimate rows for {fname}: {e}. Using default 100000")
        return 100000

# Build batch queue
batches = []
batch_files = []
batch_rows = 0

for fname in inference_files:
    file_rows = estimate_file_rows(fname, inference_data_path)
    
    if batch_rows + file_rows > MAX_ROWS_PER_BATCH:
        # Current batch would exceed limit with this file
        if batch_files and file_rows <= MAX_ROWS_PER_BATCH:
            # Normal file that doesn't fit - start new batch
            batches.append(batch_files)
            batch_files = [fname]
            batch_rows = file_rows
        else:
            # Large file (or no batch yet) - add to current batch
            batch_files.append(fname)
            batch_rows += file_rows
            # If this was a large file, close the batch for the next file to start fresh
            if file_rows > MAX_ROWS_PER_BATCH and batch_files:
                batches.append(batch_files)
                batch_files = []
                batch_rows = 0
    else:
        # File fits, add to current batch
        batch_files.append(fname)
        batch_rows += file_rows

# Add the last batch
if batch_files:
    batches.append(batch_files)

print(f"Data will be processed in {len(batches)} batch(es)")
for i, batch in enumerate(batches, 1):
    print(f"  Batch {i}: {len(batch)} file(s)")

# Initialize list to accumulate batch-level df_calc_results
all_batch_results = []
results_outfile_joint = os.path.join(save_path, f'df_calc_results_{date_time_str}.csv')

# Initialize timing variables
timeload = None
timepredict = None

# Load the previously trained models (once, before batch processing)
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

som_clf, som_model_file = load_model_by_prefix(SOMClassifier, 'som_classifier', './data/models')
mlp_clf, mlp_model_file = load_model_by_prefix(MLPClassifier, 'mlp_classifier', './data/models')

time_b = datetime.now()
timeload = time_b - timestart

# Process each batch
for batch_number, batch_file_list in enumerate(batches, 1):
    print(f'\n{"="*70}')
    print(f'BATCH {batch_number}/{len(batches)}: Processing {len(batch_file_list)} file(s)')
    print(f'{"="*70}\n')
    
    # Initialize the data manager for this batch
    fdm = FlowDataManager(
        data_file_names=batch_file_list,
        data_file_type=None,  # Is inferred from the filename ending of the 1st file
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
    # Create sample labels for each event using starting indices and batch file names
    sample_labels = []
    for i, n in enumerate(num_events):
        sample_labels.extend([batch_file_list[i]] * n)
    # Build DataFrame using all variable names
    df_calc = pd.DataFrame(no_trafo_concat, columns=var_names)
    df_calc['sample'] = sample_labels

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

    # Attach concatenated MLP predictions to df_calc
    try:
        df_calc['y_pred_mlp'] = y_pred_mlp_full
        print('updated df_calc with y_pred_mlp internally')
    except Exception as e:
        print(f'could not attach y_pred_mlp to df_calc: {e}')

    # --- Compute batch-level df_calc_results and append to joint results file
    try:
        # Ensure predictions are integers
        df_calc['y_pred_mlp'] = df_calc['y_pred_mlp'].astype(int)
        counts = df_calc.groupby('sample')['y_pred_mlp'].value_counts().unstack(fill_value=0)
        # Rename columns to mlp_{label}
        counts.columns = [f'mlp_{int(c)}' for c in counts.columns]
        df_calc_results_batch = counts.reset_index()
        
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
                df_calc_results_batch[colname] = df_calc_results_batch['sample'].map(lambda s: median_map.get(s, {}).get(colname, np.nan))

            # Reorder median columns by channel first, then population
            population_labels = sorted({int(label) for _, label in medians.index})
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
        
        # Append batch results to joint results file
        if batch_number == 1:
            # First batch: write with header
            df_calc_results_batch.to_csv(results_outfile_joint, sep=';', decimal=',', index=False)
            print(f'created df_calc_results file with batch {batch_number}')
        else:
            # Subsequent batches: append without header
            df_calc_results_batch.to_csv(results_outfile_joint, sep=';', decimal=',', index=False, mode='a', header=False)
            print(f'appended batch {batch_number} results to {results_outfile_joint}')
    except Exception as e:
        print(f'could not create batch {batch_number} df_calc_results: {e}')

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

    # Export to FCS, compensated raw data, with added columns for predictions and optional dimensionality reductions
    export_to_fcs(
        data_list=fdm.anndata_list_,  # Export the test samples
        layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
        sample_wise=False,  # Export one FCS in which the test samples are concatenated
        add_columns=add_columns,  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
        add_columns_names=add_columns_names,  # Add names for added columns
        scale_columns=add_columns_names,  # Select added columns for scaling (all that were added to the file)
        val_range=val_range,  # Range to which selected columns are scaled to
        save_path=save_path,
        save_filenames=f'inference_data_batch{batch_number}_{date_time_str}.fcs'
    )

    print(f'Batch {batch_number} processed and exported.')
    
    # Clean up memory for next batch
    print(f'Cleaning memory for batch {batch_number}...')
    del fdm, x_test, y_pred_som, y_pred_mlp, add_columns, add_columns_names
    if compute_dim_red:
        del x_som, x_tsne, x_soms_1, x_soms_2, x_tsnes_1, x_tsnes_2
    del df_calc
    import gc
    gc.collect()

# --- After all batches: Batch-level results have been appended to joint file
print(f'\n{"="*70}')
print(f'All batches processed. Population statistics saved to:')
print(f'{results_outfile_joint}')
print(f'{"="*70}\n')

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
    f.write(f'"channel cutoff for log trafo": {channel_name_to_cutoff}\n')
    f.write(f'"lin_trafo_FSSS": {lin_trafo_FSSS}\n')
    f.write (f'"dimreduction for fcs output generated": {compute_dim_red}\n')
    f.write(f'"population statistics provided in file df_calc_results_{date_time_str}.csv"\n')
    f.write(f'"events per population for all populations and samples": \n')
    f.write(f'"channel medians per population for populations {calc_populations} and channels {calcchannels}": \n')
    f.write(f'"time data load": {timeload}\n')
    f.write(f'"timetotal": {timetotal}\n')
