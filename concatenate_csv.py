# Concatenate CSV files containing different populations 
# the script expects CSV files derived from one source FCS file in a way that each population is saved...
# ...to a separate CSV file from gating in a previous step
# first part of the filenames is assumed to be the source FCS file name
# second part of the filenames is assumed to be the population name
# Example: 'Sample_ABC Bcell.csv', 'Sample_ABC Tcell.csv' and so on
# FCS file name and population name may be separated by '_' or space
# a new column indicating the population as an integer is added for later supervised training
# events occuring in two different files (assigned to two different populations by non-exclusive gating)... 
# ...will be automatically detected and the event with the higher population integer will be removed
# column 'sample_idx' is renamed from 'sample_id' if present, if not it will be created with default value 1
# CSV files from input directory are concatenated and exported as CSV (english or german style).
# an optional heatmap can be calculated 
# Last modified: 09-30-2026

import os
import sys
import yaml
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from collections import Counter
import csv
  
###      define YAML file to parse arguments     ###
with open("config_Tcell.yml", "r") as f:
    config = yaml.safe_load(f)
INPUT_DIR = config["path_concat"]
OUT = config["save_path_concat"]
CSV_out_english = config["csv_out_english"]
calc_heatmap = config["calc_heatmap"]
heatmap_channels = config["trainchannels"]
channel_name_to_cutoff = config["channel_name_to_cutoff"]

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

for fname, populations in population_dict.items():
    population_dict[fname] = {
        pop_name: population_id
        for population_id, pop_name in enumerate(sorted(populations, key=str.casefold), start=1)
    }
    for pop_name, df in dfs_all[fname].items():
        df["population"] = population_dict[fname][pop_name]

filenames_unique = set(filename_list)

print("Successfully imported: " + str([len(i) for i in dfs_all.values()]) + " populations from " + str(len(dfs_all.keys())) + " sources")
pprint({fname: dict(sorted(populations.items(), key=lambda item: item[1])) for fname, populations in population_dict.items()}, sort_dicts=False)
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

filtered_all = []

for file, concat in results.items():
    filtered, duplicates = remove_duplicates(concat)
    id_to_name = {v: k for k, v in population_dict[file].items()}
    filtered_all.append(filtered.assign(population_name=filtered["population"].map(id_to_name)))

    # --- Save filtered dataset and list of duplicates (if any)
    filtered.to_csv(os.path.join(OUT, f"{file}_concat.csv"), sep=out_sep, decimal=out_decimal, index=False)
    if duplicates.shape[0] > 0:
        print("Duplicate events found in ", file, ": ", duplicates.shape[0], " out of ", concat.shape[0], " events")
        duplicates.to_csv(os.path.join(OUT, f"{file}_duplicates.csv"), sep=out_sep, decimal=out_decimal, index=False)

    # --- Save population name to ID mapping as CSV
    with open(os.path.join(OUT, f"{file}_populations.csv"), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=out_sep)
        writer.writerow(["population_ID", "population_name"])
        for pop_name, pop_id in population_dict[file].items():
            writer.writerow([pop_id, pop_name])

if calc_heatmap:
    # --- Median per population and channel over all datasets (grouped by name, as IDs are assigned per source file)
    medians = pd.concat(filtered_all).groupby("population_name")[heatmap_channels].median()
    pop_size = pd.concat(filtered_all)["population_name"].value_counts()
    medians.index.name = "population"
    # Values below the channel cutoff are set to the cutoff before log10
    medians = np.log10(medians.clip(lower=pd.Series(channel_name_to_cutoff).reindex(medians.columns), axis=1))
    medians.insert(0, "pop_size", pop_size.reindex(medians.index))
    medians.to_csv(os.path.join(OUT, "median_per_population.csv"), sep=out_sep, decimal=out_decimal)

    # --- Heatmap with each channel scaled to 0-1
    medians_norm = medians[heatmap_channels]
    medians_norm = (medians_norm - medians_norm.min()) / (medians_norm.max() - medians_norm.min())
    plt.figure(figsize=(0.6 * len(heatmap_channels) + 3, 0.25 * len(medians) + 3))
    sns.heatmap(medians_norm, cmap="viridis", vmin=0, vmax=1, xticklabels=True, yticklabels=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "median_per_population_heatmap.png"), dpi=200)
    plt.close()

    # --- Clustermap of the same normalized values
    cm = sns.clustermap(medians_norm, cmap="viridis", vmin=0, vmax=1, xticklabels=True, yticklabels=True,
                        figsize=(0.6 * len(heatmap_channels) + 3, 0.25 * len(medians) + 3))
    cm.savefig(os.path.join(OUT, "median_per_population_clustermap.png"), dpi=200)
    plt.close(cm.fig)

print("DONE")
print("")