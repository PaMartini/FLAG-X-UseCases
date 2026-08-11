# Concatenate CSV files for supervised training
# Last modified: 11-08-2026

# Reads a list of CSV files from input directory, concatenates all into a single dataframe, checks sample and event IDs and exports as CSV.
# Input:
#   - directory containing input CSV files
# Options:
#   - define an experiment name (by default a generalized version of the sample names is used)
#   - include columns which are found in only some of the input files (by default only shared columns are exported)


import os
import sys
import argparse
from glob import glob
import pandas as pd
from pprint import pprint



### Parse arguments

parser = argparse.ArgumentParser(description="Concatenates CSV files from the same experiment into a single table for supervised training.")
parser.add_argument("-d", "--input_dir", type=str, nargs='?', default=".", help="path to directory of input files")
parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="./results", help="(optional) path to output directory (./results by default)")
parser.add_argument("-n", "--name", type=str, nargs='?', help="(optional) experiment name")
parser.add_argument("--no_filter", action='store_true', help="(optional) include only columns which are present in all of the input files")

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUT = args.output_dir
name = args.name
no_filter = args.no_filter



### Import CSV-s

path_list = glob(os.path.join(INPUT_DIR, '*.[cC][sS][vV]'))
print("")
print("Number of CSV files found in input directory: " + str(len(path_list)))

df_list = []
filename_list = []

for file in path_list:

    fname = file.split("/")[-1].split(".")[0]
    fname = fname.replace(" ", "_")
    filename_list.append(fname)
    
    df = pd.read_csv(file, decimal=".", sep=",", low_memory=False)

    # Add column for sample name
    df.insert(0, "Sample", fname)

    df_list.append(df)

print("Successfully imported: " + str(len(df_list)))
print("")



### Concatenate

# --- Check if columns are uniform across all dfs
cols_same = True
dfs_concat = []
dfs_exclude = []

for df in df_list[1:]:
    if list(df_list[0].columns) == list(df.columns):
        dfs_concat.append(df)
    else:
        cols_same = False
        dfs_exclude.append(df)

# --- Concatenate dfs
result = None
if len(dfs_exclude) == 0 or no_filter:
    result = pd.concat(df_list)
else:
    result = pd.concat(dfs_concat)
    print("--> Warning: some of the input files have additional columns not present in others.")
    print("--> By default such columns are excluded from the result. To include them, run the script with option --no_filter.")
    



### Export result

print("Exporting result to: " + OUT)
if name is None:
    name = max(filename_list, key=len)
    for fn in filename_list:
        for i in range(len(fn)):
            if max(filename_list, key=len)[i] != fn[i]:
                name = name[:i] + "x" + name[(i+1):]

result.to_csv(os.path.join(OUT, f"concat_{name}.csv"), sep=",", decimal=".", index=False)

print("DONE")
print("")