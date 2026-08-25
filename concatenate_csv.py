# Concatenate CSV files for supervised training
# Last modified: 25-08-2026

# Reads a list of CSV files from input directory, concatenates all into a single dataframe and exports as CSV.
# Input (read from YAML config file):
#   - directory containing input CSV files


import os
import sys
import yaml
from glob import glob
import pandas as pd
from pprint import pprint
import csv



### Parse arguments

with open("config_Bcell.yml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = config["save_path_unsup_training"]
OUT = config["path_sup_training"]



### Import CSV-s

path_list = glob(os.path.join(INPUT_DIR, '*.[cC][sS][vV]'))
print("")
print("Number of CSV files found in input directory: " + str(len(path_list)))

df_list = []
filename_list = []
population_dict = {}

for file in sorted(path_list):

    # Detect CSV formatting style (assume European by default)
    format_en = False
    with open(file, 'r') as f:
        first_line = f.readline()
        if first_line.find(",") != -1:
            format_en = True
        
    # Read in the file
    if format_en:
        df = pd.read_csv(file, decimal=".", sep=",", low_memory=False)
    else:
        df = pd.read_csv(file, decimal=",", sep=";", low_memory=False)

    # Get sample name, add as column if no sample_id column present
    fname = file.split("/")[-1].split(".")[0]
    fname = fname.replace(" ", "_")
    filename_list.append(fname)
    if "sample_ID" not in df.columns:
        df.insert(0, "sample_name", fname)

    # Get population name, add as column
    pop_name = fname.split("_")[-1]   # assume the last part of the file name to be the population name
    if pop_name not in population_dict:
        population_dict[pop_name] = len(population_dict)
    df.insert(len(df.columns), "population", population_dict[pop_name])

    df_list.append(df)

print("Successfully imported: " + str(len(df_list)))
print("")



### Concatenate

# --- Check if columns are uniform across all dfs
dfs_concat = [df_list[0]]
cols_same = True

for df in df_list[1:]:
    if list(df_list[0].columns) == list(df.columns):
        dfs_concat.append(df)
    else:
        cols_same = False

# --- Concatenate dfs
result = pd.concat(dfs_concat)
if not cols_same:
    print("--> Warning: some of the input files have missing or additional columns, these files are excluded from the result.")
    



### Export result

print("Exporting results to: " + OUT)

fn_roots = [fn.rsplit("_", 1)[0] for fn in filename_list]
name = max(fn_roots, key=len)
for fn in fn_roots:
    for i in range(len(fn)):
        if max(fn_roots, key=len)[i] != fn[i]:
            name = name[:i] + "x" + name[(i+1):]

result.to_csv(os.path.join(OUT, f"{name}_concat.csv"), sep=",", decimal=".", index=False)

# Save population name to ID mapping as CSV
with open(os.path.join(OUT, f"{name}_populations.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["population_ID", "population_name"])
    for pop_name, pop_id in population_dict.items():
        writer.writerow([pop_id, pop_name])

print("DONE")
print("")