# tested and running 2026-07-13
# checks input csv without changes
# !! set the following before start: !!
# german or english type of csv (decimal, sep), line 21
# determine training or testing folder to be analyzed
# delete any datafiles besides target csv from folder 


import os
import pandas as pd
from datetime import datetime

timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%H-%M")

csv_path = './data/training' # change to training or testing depending on the csv you want to check (also line 17)
csv_out = './results'
input_list = os.listdir(csv_path)
print(input_list)
for file in input_list:
    df = pd.read_csv(os.path.join(csv_path, file), decimal=",", sep=";") # change as needed
    input_channels = df.columns.tolist()
    with open(os.path.join(csv_out, f'csv_training_{date_time_str}.txt'), 'a') as f:
        f.write(file + "\n")
        f.write(str(input_channels) + "\n")