# CSV preprocessing
# Last modified: 11-08-2026

# Reads a list of CSV files from input directory, checks the formatting and exports reformatted CSV-s into output directory.
# Input:
#   - path to input dir
# Options:
#   - specify path to output directory (default is './results')
#   - check channel names against list of expected labels; if no list is provided, the most commonly occurring name for each channel is used
#   - add unique event ID-s



import os
import argparse
from glob import glob
import pandas as pd
from pprint import pprint
from collections import Counter




### Parse arguments

parser = argparse.ArgumentParser(description="Checks CSV files for expected formatting and reformats them in a uniform style.")
parser.add_argument("-d", "--input_dir", type=str, nargs='?', default=".", help="path to directory of input files")
parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="./results", help="(optional) path to output directory (./results by default)")
parser.add_argument("-id", "--add_IDs", action='store_true', help="(optional) add unique event identifiers")
parser.add_argument("-en", "--format_en", action='store_true', help="(optional) use if input CSV files are formatted in English style (German is assumed by default)")
parser.add_argument("-c", "--channels", type=str, help="(optional) path to TXT file containing expected channel names")

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUT = args.output_dir
add_IDs = args.add_IDs
format_en = args.format_en
exp_channels = args.channels





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
    
    # Read in the file
    if format_en:
        df = pd.read_csv(file, decimal=".", sep=",", low_memory=False)
    else:
        df = pd.read_csv(file, decimal=",", sep=";", low_memory=False)

    # Remove unnecessary column if present
    df = df.drop("FS PEAK", axis=1, errors="ignore")

    # Add unique event IDs
    if add_IDs:
        IDs = [("ID_" + str(id).zfill(len(str(df.shape[0])))) for id in range(0, df.shape[0])]
        df.insert(0, "Event ID", IDs, allow_duplicates=False)

    # Add df to list
    df_list.append(df)

print("Successfully imported: " + str(len(df_list)))
print("")




### Check channel names

# --- Compare colnames across all imported df-s
# --- Use channel name list from input if provided, otherwise use most common name per column

if exp_channels is not None:

    # Read in channel names from TXT
    with open(exp_channels, 'r') as file:
        channels = file.read().splitlines()
    
    if len(channels) == 1:
        if channels.find(",") == -1:
            channels = channels.split(";")
        else:
            channels = channels.split(",")

        for ch in channels:
            if ch[0] == " ":
                ch = ch[1:]

    print("Expected channels from list (" + str(len(channels)) + "): " + str(channels))
    print("")

    # Rename columns using provided channel names
    try:
        for df in df_list:
            df.columns = ["Event ID"] + channels
    except ValueError as err:
       print(err)
       print("--> Number of channels in TXT input doesn't match number of CSV columns. Please check the input files.")
       print("")
else:
    colnames_in = pd.DataFrame([list(df.columns) for df in df_list])
    colnames_ok = []
    colnames_check = []
    colnames_final = []

    # Correct for typos: if multiple names for same column are present, use the most commonly occurring one
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