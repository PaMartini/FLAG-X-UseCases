# list csv and change format from german to english. 
# deletes FS PEAK column if exists 
# renames columns to final_channels list to avoid spelling errors.
# delete any data besides target csv from folder before running
# delete input csv after run and work with new "corr" files
# run correctly 2026-03-25

import os
import pandas as pd
from datetime import datetime

timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%H-%M")

csv_path = './data/training'
final_channels = ['FS INT', 'SS INT', '15-FITC', '13-PE', '34-ECD', '117-PC5.5', '33-PC7', '2-APC', '7-APC-AF700', 'APC-AF750', 'HLADR-PB', '45-CO', 'TIME']
input_list = os.listdir(csv_path)

for file in input_list:
    df = pd.read_csv(os.path.join(csv_path, file), decimal=",", sep=";")
    input_channels = df.columns.tolist() # use original column names for description file
    df = df.drop(['FS PEAK'], axis=1, errors='ignore')
    df.columns = final_channels
    df.to_csv(os.path.join(csv_path, f"corr_{file}"), sep=",", decimal=".", index=False)
    with open(os.path.join(csv_path, f'csv_description_{date_time_str}.txt'), 'a') as f:
        f.write(file + "\n")
        f.write(str(input_channels) + "\n")
with open(os.path.join(csv_path, f'csv_description_{date_time_str}.txt'), 'a') as f:
    f.write("final channels: " + "\n")
    f.write(str(final_channels) + "\n")