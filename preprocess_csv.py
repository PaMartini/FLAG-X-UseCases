# CSV preprocessing
# Last modified: 25-08-2026

# Reads a list of CSV files from input directory, checks the formatting and exports reformatted CSV-s into output directory.
# Input (read from YAML config file):
#   - path to input dir
#   - list of expected channel names



import os, sys
import yaml
from glob import glob
import pandas as pd
from pprint import pprint
from collections import Counter




### Parse arguments

with open("config_Bcell.yml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["path_sup_training"]
OUT = config["save_path_sup_training"]
channels = config["trainchannels"]




### Import and reformat CSV-s

# --- Get list of paths to input CSV-s
path_list = glob(os.path.join(INPUT_DIR, '*.[cC][sS][vV]'))
print("")
print("Number of CSV files found in input directory: " + str(len(path_list)))

# --- Read CSV-s, reformat and add event IDs
df_list = []
filename_list = []

for file in path_list:

    fname = file.split("/")[-1].split(".")[0]
    fname = fname.replace(" ", "_")
    filename_list.append(fname)

    # Detect CSV formatting style
    # --- assume European formatting by default (decimal point = ",", separator = ";")
    format_en = False
    # --- if the first line contains a comma, assume English formatting (decimal point = ".", separator = ",")
    with open(file, 'r') as f:
        first_line = f.readline()
        if first_line.find(",") != -1:
            format_en = True
    
    # Read in the file
    if format_en:
        df = pd.read_csv(file, decimal=".", sep=",", low_memory=False)
    else:
        df = pd.read_csv(file, decimal=",", sep=";", low_memory=False)

    # Remove unnecessary column if present
    df = df.drop("FS PEAK", axis=1, errors="ignore")

    # Add unique event IDs (int)
    df.insert(0, "Event ID", range(0, df.shape[0]), allow_duplicates=False)

    # Add df to list
    df_list.append(df)

print("Successfully imported: " + str(len(df_list)))
print("")




### Check channel names

# --- Compare colnames across all imported df-s
# --- Use channel name list from input if provided, otherwise use most common name per column

# Rename columns using provided channel names
print("Expected channels (" + str(len(channels)) + "): " + str(channels))
print("")
if channels is not None and len(channels) == df.shape[1] - 1:
    for df in df_list:
        df.columns = ["Event ID"] + channels
else:
    # If no list is provided or the length doesn't match, use the most common name per column (correcting for typos)
    print("No channel name list provided or length doesn't match number of columns. Using most common name per column instead.")
    colnames_in = pd.DataFrame([list(df.columns) for df in df_list])
    colnames_ok = []
    colnames_check = []
    colnames_final = []

    for i in list(colnames_in.columns):
        col = colnames_in[i]
        if len(col.unique()) == 1:
            colnames_ok.append(col.unique()[0])
            colnames_final.append(col.unique()[0])
        else:
            colnames_check.append(list(col.unique()))
            if Counter(col).most_common()[0][0] == None:
                colnames_final.append(Counter(col).most_common()[1][0])
            else:
                colnames_final.append(Counter(col).most_common()[0][0])
            
    print("Column names detected (" + str(df.shape[1] - 1) + "): " + str(colnames_ok[1:]))   # don't count event ID column
    if len(colnames_check) > 0:
        print("The following column names were ambiguous and will be replaced with the most common version:")
        pprint(colnames_check)
    print("")




### Export results

print("Exporting results to: " + OUT)
for i, df in enumerate(df_list):
    df.to_csv(os.path.join(OUT, f"corr_{filename_list[i]}.csv"), sep=",", decimal=".", index=False)
print("DONE")
print("")