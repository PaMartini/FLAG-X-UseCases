# list csv and change format from german to english. run correctly 2026-01-31
import os
import pandas as pd

# working for only one folder at a time
# delete any data besides target csv from folder before running
# delete input csv after run and work with new files
csv_path = './data/training'
input_list = os.listdir(csv_path)
print(input_list)
for file in input_list:
    df = pd.read_csv(os.path.join(csv_path, file), decimal=",", sep=";")
    input_channels = df.columns.tolist()
    df.to_csv(os.path.join(csv_path, f"new_{file}"), sep=",", decimal=".", index=False)
    with open(os.path.join(csv_path, 'csv_description.txt'), 'a') as f:
        f.write(file + "\n")
        f.write(str(input_channels) + "\n")