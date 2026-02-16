# tested and running 2026-02-12
# check_input csv without changes
# if necessary, transfrom to english type of csv beforehand
# delete any data besides target csv from folder before running
# working for only one folder at a time

import os
import pandas as pd

csv_path = './data/testing' # change to training or testing depending on the csv you want to check (also line 17)
csv_out = './results'
input_list = os.listdir(csv_path)
print(input_list)
for file in input_list:
    df = pd.read_csv(os.path.join(csv_path, file), decimal=".", sep=",")
    input_channels = df.columns.tolist()
    with open(os.path.join(csv_out, 'csv_testing.txt'), 'a') as f:
        f.write(file + "\n")
        f.write(str(input_channels) + "\n")