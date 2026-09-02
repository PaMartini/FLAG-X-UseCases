# Data preprocessing
# Last modified: 01-09-2026

# Reads a list of FCS or CSV files from input directory, checks the formatting and exports reformatted data into output directory as CSV.
# Input (read from YAML config file):
#   - path to input and output dir
#   - list of expected channel names and channels to be later used for training



import os
import yaml
import pandas as pd
from pprint import pprint
from flagx.io import FlowDataManager




### Parse arguments

with open("config_Bcell.yml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["path_sup_training"]
OUT = config["path_sup_training"]
markers = config["marker_channels"]
trainchannels = config["trainchannels"]




### Import data
### (if a file is found in multiple formats, FCS will be preferred)

# --- Get list of paths to input files
file_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.csv', '.CSV', '.fcs', '.FCS'))])
path_list = [os.path.join(INPUT_DIR, f) for f in file_list]

print("")
print("Importing files...")



dfs = {}

# --- FCS import
if len([f for f in file_list if f.endswith(('.fcs', '.FCS'))]) != 0:
    
    # Initialize FlowDataManager & load data into memory
    fdm = FlowDataManager(data_file_names = [f for f in file_list if f.endswith(('.fcs', '.FCS'))],
                    data_file_path = INPUT_DIR,
                    save_path = OUT,
                    verbosity = 1)
    fdm.load_data_files_to_anndata()

    # Convert to dataframe
    for adata in fdm.anndata_list_:
        fname = adata.uns["filename"].split(".")[0]
        dfs[fname] = adata.to_df()

# --- CSV import
if len([f for f in file_list if f.endswith(('.csv', '.CSV'))]) != 0:
    
    # Find files to be imported from CSV - only if not already imported from FCS (to avoid duplicates)
    import_from_csv = [f for f in path_list if f.endswith(('.csv', '.CSV')) and f.split("/")[-1].split(".")[0] not in dfs.keys()]

    # --- Check formatting style and import
    if len(import_from_csv) != 0:
        for file in import_from_csv:
            with open(file, 'r') as f:
                first_line = f.readline()
                fname = file.split("/")[-1].split(".")[0]
                if first_line.count(",") > first_line.count(";"):
                    dfs[fname] = pd.read_csv(file, decimal=".", sep=",")
                else:
                    dfs[fname] = pd.read_csv(file, decimal=",", sep=";")

print("")
print("Dataframes successfully imported: " + str(len(dfs)))
pprint(list(dfs.keys()))
print("")



### Preprocess dataframes

# --- Remove unnecessary column if present

for df in dfs.values():
    df = df.drop("FS PEAK", axis=1, errors="ignore")


# --- Sync column names

print("Syncing column names...")
print("")

scatter = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
expected_channels = scatter + [m for m in markers if m not in scatter]
exp_ch_ignorecase = [ch.casefold() for ch in expected_channels]

all_cols_check = {}
rename_cols = {}

for file, df in dfs.items():

    # Find columns not in list of expected channels
    cols_ignorecase = [ch.casefold() for ch in df.columns]
    cols_same_ignorecase = [ch.casefold() for ch in expected_channels if ch.casefold() in list(set(exp_ch_ignorecase) & set(cols_ignorecase))]
    cols_check = [ch for ch in df.columns if ch.casefold() not in cols_same_ignorecase]
    if len(cols_check) > 0:
        all_cols_check[file] = cols_check

    # Correct for uppercase/lowercase differences
    cols_to_rename = {}
    for col in df.columns:
        if col not in expected_channels and col.casefold() in exp_ch_ignorecase:
            cols_to_rename[col] = expected_channels[exp_ch_ignorecase.index(col.casefold())]
    if len(cols_to_rename) > 0:
        rename_cols[file] = cols_to_rename

if len(all_cols_check) > 0:
    print("Column names not found in the list of expected channels (please check input files):")
    for file, cols in all_cols_check.items():
        print(f' --- {file}:')
        print(f'     {cols}')
    print("")

if len(rename_cols) > 0:
    print("Changes made:")
    for f, d in rename_cols.items():
        changes = [old + " -> " + new for old, new in d.items()]
        print(f' --- {f}: {changes}')
    print("")


# --- Find dataframes with missing columns

missing_cols = {
    "info": {},
    "warning": {}
}

for file, df in dfs.items():
    missing_channels = [ch for ch in expected_channels if ch not in df.columns]
    if len(missing_channels) > 0:
        missing_cols["info"][file] = missing_channels
    missing_trainchannels = [ch for ch in trainchannels if ch not in df.columns]
    if len(missing_trainchannels) > 0:
        missing_cols["warning"][file] = missing_trainchannels

if len(missing_cols["warning"]) + len(missing_cols["info"]) != 0:
    print(f'!! Warning: the following dataframes are missing expected channels:')
    for file in dfs.keys():
        if file in missing_cols["warning"].keys() or file in missing_cols["info"].keys():
            print(f' --- {file}:')
        if file in missing_cols["warning"].keys():
            print(f'     ... does not include training channels: {missing_cols["warning"][file]}')
        if file in missing_cols["info"].keys():
            print(f'     ... does not include channels (not used for training): {[c for c in missing_cols["info"][file] if c not in trainchannels]}')
    print("")


# --- Add unique event IDs

print("Adding event IDs...")
print("")

for df in dfs.values():
    # --- Add unique event IDs
    df.insert(0, "Event ID", range(0, df.shape[0]), allow_duplicates=False)



### Export results

print("Exporting results to: " + OUT)
for file, df in dfs.items():
    fname = "corr_" + file + ".csv"
    df.to_csv(os.path.join(OUT, fname.replace(" ", "_")), sep=",", decimal=".", index=False)
print("DONE")
print("")