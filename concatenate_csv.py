# Concatenate CSV files for supervised training
# Last modified: 02-09-2026

# Reads a list of CSV files from input directory, concatenates all into a single dataframe and exports as CSV.
# Input (read from YAML config file):
#   - directory containing input CSV files


import os
import sys
import yaml
from glob import glob
import pandas as pd
from pprint import pprint
from collections import Counter
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

dfs_all = {}
filename_list = []
population_dict = {}


for file in sorted(path_list):

    # Get source file and population name
    fname = file.split("/")[-1].split(".")[0]
    pop_name = fname.split("_")[-1]   # assume the last part of the file name to be the population name
    fname = fname[0:fname.rfind("_")]
    if fname not in filename_list:
        filename_list.append(fname) 
    if fname not in population_dict.keys():
        population_dict[fname] = {}
    if pop_name not in population_dict[fname].keys():
        population_dict[fname].update({pop_name : len(population_dict[fname])})

    # Detect CSV formatting style (assume European by default)
    format_en = False
    with open(file, 'r') as f:
        first_line = f.readline()
        if first_line.count(",") > first_line.count(";"):
            format_en = True
        
    # Read in the file
    if format_en:
        df = pd.read_csv(file, decimal=".", sep=",", low_memory=False)
    else:
        df = pd.read_csv(file, decimal=",", sep=";", low_memory=False)

    # Add population column
    df.insert(len(df.columns), "population", population_dict[fname][pop_name])

    if fname not in dfs_all:
        dfs_all[fname] = {pop_name : df}
    else:
        dfs_all[fname].update({pop_name: df})

filenames_unique = set(filename_list)

print("Successfully imported: " + str([len(i) for i in dfs_all.values()]) + " populations from " + str(len(dfs_all.keys())) + " sources")
pprint(population_dict)
print("")



## Concatenate

results = dict(zip(list(dfs_all.keys()), [None] * len(dfs_all)))

# --- Check if columns are uniform across all dfs from same source
cols_first = dict(zip(list(dfs_all.keys()), [None] * len(list(dfs_all.keys()))))
for f, c in cols_first.items():
    c = list(list(dfs_all[f].values())[0].columns)
    cols_first[f] = c

concat_warnings = dict(zip(list(dfs_all.keys()), [None] * len(list(dfs_all.keys()))))
for file, pops in dfs_all.items():
    cols_same = []
    to_concat = []
    for df in list(pops.values()):
        if list(df.columns) == cols_first[file]:
            cols_same.append(True)
            to_concat.append(df)
        else:
            cols_same.append(False)
    results[file] = pd.concat(to_concat)
    if False in cols_same:
        concat_warnings[file] = "col_mismatch"


# --- Remove duplicate events
def remove_duplicates(res):
    result_dupl = res.sort_values("population")
    result_dupl["duplicated"] = result_dupl.duplicated(subset=["Event ID"])
    result_dupl = result_dupl[result_dupl["duplicated"] == True]
    result_filtered = res.sort_values("population").drop_duplicates("Event ID").sort_index()
    return [result_filtered, result_dupl]



### Export result

print("Exporting results to: " + OUT)

for file, concat in results.items():

    # --- Save filtered dataset and list of duplicates (if any)
    remove_duplicates(concat)[0].to_csv(os.path.join(OUT, f"{file}_concat.csv"), sep=",", decimal=".", index=False)
    if (remove_duplicates(concat)[1].shape[0] > 0):
        print("Duplicate event IDs found in ", file, ": ", remove_duplicates(concat)[1].shape[0], " out of ", concat.shape[0], " events")
        print(remove_duplicates(concat)[1]["Event ID"])
        remove_duplicates(concat)[1].to_csv(os.path.join(OUT, f"{file}_duplicates.csv"), sep=",", decimal=".", index=False)

    # --- Save population name to ID mapping as CSV
    with open(os.path.join(OUT, f"{file}_populations.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["population_ID", "population_name"])
        for pop_name, pop_id in population_dict[file].items():
            writer.writerow([pop_id, pop_name])

print("DONE")
print("")