# open TSNE test remap
# remap new FCM csv to TSNE embedding
# parameters must be similar to original embedding
# tested and running 2026-06-08

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from openTSNE import TSNE
print('loading packages and paths...')
timestart = datetime.now()
date_time_str = timestart.strftime("%Y-%m-%d_%Hh%M")   

# --- Define selected Parameters for the workflow ------------------------------------
trainchannels = [
     'FS INT', 'SS INT', '15-FITC', '13-PE', '33-PC7', '2-APC', '7-APC-AF700',
     '34-ECD', '117-PC5.5', 'HLADR-PB', '45-CO'
] # List of channels to be used for training/testing. Check spelling and consistency across samples. Adjust if needed.

# Define path to training data
testing_data_path = './data/testing'
# Get list of files in the data directory (only include ones ending with .csv)
testing_files = sorted([fn for fn in os.listdir(testing_data_path) if fn.endswith('.csv')])
# Load the testing files into pandas dataframes
testing_data_dfs_complete = [pd.read_csv(os.path.join(testing_data_path, fn)) for fn in testing_files]
testing_set_df = testing_data_dfs_complete[0].loc[:, trainchannels]
# transform to numpy array for TSNE and apply arcsinh transformation
testing_set = testing_set_df.to_numpy()
testing_set_t = np.arcsinh(testing_set / 300)

save_path = './results/TSNE-test'

tsne_model = pickle.load(open(os.path.join(save_path, 'tsne_model.pkl'), 'rb'))
x_tsne = pickle.load(open(os.path.join(save_path, 'tsne_embedding.pkl'), 'rb'))

y_tsne = x_tsne.transform(testing_set_t)
# x_tsne = tsne_model.fit(training_set_t)

fig = plt.figure(dpi=300)
plt.scatter(y_tsne[:, 0], y_tsne[:, 1], s=1, alpha=0.5)
fig.savefig(os.path.join(save_path, f'tsne_testing_{date_time_str}.png'))
plt.close(fig)