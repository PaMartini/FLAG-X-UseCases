# Data preprocessing
# Last modified: 09-09-2026

# Reads a list of FCS, LMD or CSV files from input directory, checks the formatting and performs compensation of raw data (if applicable).

# Input (read from YAML config file):
#   - 'path_sup_training': path to directory containing the input files, output will be saved here as well
#   - 'all_channels': list of expected channel names
#   - 'delete_channels': channels to be removed if found in input

# Output options (to be set in the YAML config file):
#   - 'save_as_csv': if False, results are saved as FCS; if set to True, use 'csv_out_english' to set European (default) or English style formatting
#   For FCS output:
#      - 'apply_comp': whether to apply compensation; if no compensation matrix is found, a warning will be printed
#      - 'fcs_export_raw': if set to True, both the compensated data and the raw data with compensation matrix are saved (as separate files); if False, only the compensated data is exported




import os
import yaml
from flagx.io import FlowDataManager, export_to_fcs
from anndata import AnnData
from typing import List, Tuple
from datetime import datetime


def find_unique_input(files: List[str], format_priority: List[str]) -> List[str]:
    """Check if any files are found in multiple formats in the input directory, if yes choose based on given order of preference."""

    keep = files.copy()
    prio = {ext: i for i, ext in enumerate(format_priority, start=1)}

    if len(files) != len(set([f.split(".")[0] for f in files])):
        input_sorted = {}

        for file in files:
            fname = file.split(".")[0]
            fformat = file.split(".")[-1].casefold()
            if fname not in input_sorted.keys():
                input_sorted[fname] = fformat
            else:
                if prio[fformat] < prio[input_sorted[fname]]:
                    input_sorted[fname] = fformat

        keep = [f"{k}.{v}" for k, v in input_sorted.items()]

    return keep


def import_data(fpath: str, input_dir: str, comp: bool) -> Tuple[AnnData, str, str, bool]:
    """Import data from FCS, LMD or CSV format, apply compensation if available (unless otherwise specified), and return the result as an AnnData object."""

    file = fpath.split("/")[-1]
    fformat = file.split(".")[-1].casefold()
    fname = file.split(".")[0]
    comp_warn = False
    
    # Initialize FlowDataManager
    fdm = FlowDataManager(data_file_names = [file],
                          data_file_path = input_dir,
                          verbosity = 1)
        
    if fformat == 'csv':
        # Detect formatting style based on first line and import CSV accordingly
        with open(fpath, 'r') as file:
            first_line = file.readline()
            fname = fpath.split("/")[-1].split(".")[0]
            if first_line.count(",") > first_line.count(";"):
                read_csv_kwargs = {"decimal" : ".", "sep" : ","}
            else:
                read_csv_kwargs = {"decimal" : ",", "sep" : ";"}
        # Import data
        fdm.load_data_files_to_anndata(read_csv_kwargs = read_csv_kwargs)
            
    if fformat in ['fcs', 'lmd']:
        # Import data
        fdm.load_data_files_to_anndata()
        # Apply compensation unless otherwise specified
        if fdm.anndata_list_[0].uns['meta']['spill'] is None:
            comp_warn = True
        if comp and not comp_warn:
            fdm.sample_wise_compensation()
    
    # Return imported file as AnnData object
    return fdm.anndata_list_[0], file, fname, comp_warn


def drop_channels(adata: AnnData, cols_del: List[str]) -> Tuple[AnnData, list]:
    """Check the variables of an AnnData object against an input list of names, drop the ones that are found in the list, and return the resulting subset and a list of the dropped variables."""
    cols_dropped = [col for col in adata.var_names if col in cols_del]
    cols_to_keep = [col for col in adata.var_names if col not in cols_del]
    return adata[:, cols_to_keep].copy(), cols_dropped


def check_channels(adata: AnnData, channels: List[str]) -> Tuple[AnnData, dict]:
    """Check the variables of an AnnData object against an input list of names, list additional or missing variables and correct any uppercase/lowercase differences."""

    # Compare columns with list of expected channel names
    result = {}
    result["cols_in"] = list(adata.var_names)
    result["cols_missing"] = [col for col in channels if col not in adata.var_names]
    result["cols_extra"] = [col for col in adata.var_names if col not in channels]
    result["cols_renamed"] = {}

    # Check and correct for uppercase/lowercase differences
    new_names = []
    ch_ignorecase = [ch.casefold() for ch in channels]
    for col in adata.var_names:
        if col not in channels and col.casefold() in ch_ignorecase:
            result["cols_renamed"][col] = channels[ch_ignorecase.index(col.casefold())]
            new_names.append(channels[ch_ignorecase.index(col.casefold())])
        else:
            new_names.append(col)

    adata_new = adata.copy()
    adata_new.var_names = new_names

    return adata_new, result


def add_event_IDs(adata: AnnData) -> Tuple[AnnData, bool]:
    """Add an Event ID column if not already present."""

    added_IDs = False
    if 'Event ID' not in adata.var_names and 'Event ID' not in adata.obs:
        adata.obs['Event ID'] = range(adata.n_obs)
        added_IDs = True

    return adata, added_IDs




### Get config params

with open("config_Bcell.yml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["path_sup_training"]
OUT = config["path_sup_training"]
CHANNELS = config["all_channels"]
COLS_DEL = config["delete_channels"]
CSV_OUT_EN = config["csv_out_english"]
FCS_OUT_RAW = config["export_raw_fcs"]
APPLY_COMP = config["apply_comp"]
SAVE_AS_CSV = config["save_as_csv"]
IMPORT_PRIO = config["import_format_priority"]



### Get list of paths to input files

file_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.csv', '.CSV', '.fcs', '.FCS', '.lmd', '.LMD'))])
path_list = [os.path.join(INPUT_DIR, f) for f in find_unique_input(file_list, IMPORT_PRIO)]



### Create TXT out file

txt_out = os.path.join(OUT, f'preprocess_data_{datetime.now().strftime("%d-%m-%Y_%H%M%S")}_out.txt')
with open(txt_out, "w") as f:
    f.write(f"Import directory: {INPUT_DIR}\n")
    f.write(f"Compatible files found: {len(path_list)}\n")
    f.write("\n")



### Perform steps

imports = 0
for fpath in path_list:

    # --- Import data
    adata, file, fname, comp_warn = import_data(fpath, INPUT_DIR, APPLY_COMP)
    imports += 1

    # --- Preprocessing steps
    adata, cols_dropped = drop_channels(adata, COLS_DEL)
    adata, colname_changes = check_channels(adata, CHANNELS)
    adata, added_IDs = add_event_IDs(adata)

    # --- Export results
    ids = 'Event ID' in adata.obs
    if SAVE_AS_CSV:
        df = adata.to_df().join(adata.obs['Event ID']) if ids else adata.to_df()
        df.to_csv(os.path.join(OUT, f'corr_{fname}.csv'),
                  sep="," if CSV_OUT_EN else ";",
                  decimal="." if CSV_OUT_EN else ",",
                  index=False)
    else:
        # Export .X matrix
        export_to_fcs(data_list = [adata],
                      add_columns = [adata.obs['Event ID']] if ids else None,
                      add_columns_names = ['Event ID'] if ids else None,
                      sample_wise = True,
                      save_path = OUT,
                      save_filenames = [f'corr_{fname}.fcs'])
        files_exp = [f'corr_{fname}.fcs']
        # Additionally if specified: export uncompensated data
        if FCS_OUT_RAW and 'uncompensated' in adata.layers:
            export_to_fcs(data_list = [adata],
                          layer_key = 'uncompensated',
                          add_columns = [adata.obs['Event ID']] if ids else None,
                          add_columns_names = ['Event ID'] if ids else None,
                          sample_wise = True,
                          save_path = OUT,
                          save_filenames = [f'corr_{fname}_uncompensated.fcs'])
            files_exp.append(f'corr_{fname}_uncompensated.fcs')
            # Save compensation matrix separately
            adata.uns['meta']['spill'].to_csv(os.path.join(OUT, f'corr_{fname}_compMatrix.csv'),
                                              sep = "," if CSV_OUT_EN else ";",
                                              decimal = "." if CSV_OUT_EN else ",",
                                              index = True,
                                              index_label = "id")
            files_exp.append(f'corr_{fname}_compMatrix.csv')

    # --- Add info & changes to out file
    with open(txt_out, "a") as f:
        f.write(f'> {file}:\n')
        f.write(f" --- Columns read: {', '.join(colname_changes['cols_in'])}\n")
        f.write(f" --- Columns dropped: {', '.join(cols_dropped) if len(cols_dropped) > 0 else 'none'}\n")
        f.write(f" --- Column names updated: {', '.join([old + ' -> ' + new for old, new in colname_changes['cols_renamed'].items()]) if len(colname_changes['cols_renamed']) > 0 else 'none'}\n")
        f.write(f" --- Event ID column: {'added' if added_IDs else 'already present'}\n")
        if SAVE_AS_CSV:
            f.write(f" --- Files exported: {f'corr_{fname}.csv'}\n")
            f.write(f" --- CSV formatting style: {'EN' if CSV_OUT_EN else 'DE'}")
        else:
            f.write(f" --- Compensation applied: {'no' if comp_warn or not APPLY_COMP else 'yes'}\n")
            f.write(f" --- Files exported: {files_exp}\n")
        f.write("\n")


with open(txt_out, "a") as f:
    f.write(f"Total files successfully imported: {imports}\n")
    f.write(f"Results saved to: {OUT}")