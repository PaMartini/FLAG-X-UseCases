# Concatenate CSV files containing different populations 
# the script expects CSV files derived from one source FCS file in a way that each population is saved...
# ...to a separate CSV file from gating in a previous step
# first part of the filenames is assumed to be the source FCS file name
# second part of the filenames is assumed to be the population name
# Example: 'Sample_ABC Bcell.csv', 'Sample_ABC Tcell.csv' and so on
# FCS file name and population name may be separated by '_' or space
# a new column indicating the population is added for later supervised training
# events occuring in two different files (assigned to two different populations by non-exclusive gating)... 
# ...will be automatically detected and the event with the higher population value will be removed
# CSV files from input directory are concatenated and exported as CSV (english or german style).
# Last modified: 05-09-2026

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
INPUT_DIR = config["path_concat"]
OUT = config["save_path_concat"]
CSV_out_english = config["csv_out_english"]

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
    fname = fname.replace(" ", "_")
    pop_name = fname.split("_")[-1]   # assume the last part of the file name to be the population name
    fname = fname[0:fname.rfind("_")]
    if fname not in filename_list:
        filename_list.append(fname) 
    if fname not in population_dict.keys():
        population_dict[fname] = {}
    if pop_name not in population_dict[fname].keys():
        population_dict[fname].update({pop_name : len(population_dict[fname]) + 1})

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

    # make sure sample identifier column is available (necessary for supervised training)
    if "sample_idx" not in df.columns:
        if "sample_id" in df.columns:
            df = df.rename(columns={"sample_id": "sample_idx"})
        else:
            df["sample_idx"] = 1
    df["sample_idx"] = df.pop("sample_idx")

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
    # Prefer "Event ID" for duplicate detection, fall back to comparing all columns except "population" (which is only added during concatenation)
    dupl_subset = ["Event ID"] if "Event ID" in res.columns else [c for c in res.columns if c != "population"]
    result_dupl = res.sort_values("population")
    result_dupl["duplicated"] = result_dupl.duplicated(subset=dupl_subset)
    result_dupl = result_dupl[result_dupl["duplicated"] == True]
    result_filtered = res.sort_values("population").drop_duplicates(subset=dupl_subset).sort_index()
    return [result_filtered, result_dupl]

### Export result
print("Exporting results to: " + OUT)

out_sep = "," if CSV_out_english else ";"
out_decimal = "." if CSV_out_english else ","

for file, concat in results.items():

    # --- Save filtered dataset and list of duplicates (if any)
    remove_duplicates(concat)[0].to_csv(os.path.join(OUT, f"{file}_concat.csv"), sep=out_sep, decimal=out_decimal, index=False)
    if (remove_duplicates(concat)[1].shape[0] > 0):
        print("Duplicate events found in ", file, ": ", remove_duplicates(concat)[1].shape[0], " out of ", concat.shape[0], " events")
        remove_duplicates(concat)[1].to_csv(os.path.join(OUT, f"{file}_duplicates.csv"), sep=out_sep, decimal=out_decimal, index=False)

    # --- Save population name to ID mapping as CSV
    with open(os.path.join(OUT, f"{file}_populations.csv"), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=out_sep)
        writer.writerow(["population_ID", "population_name"])
        for pop_name, pop_id in population_dict[file].items():
            writer.writerow([pop_id, pop_name])

print("DONE")
print("")