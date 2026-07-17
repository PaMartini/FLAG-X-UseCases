# open TSNE test
# map FCM csv to TSNE and save embedding (for remapping of a different file)
# simple grafics included
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
trainchannels = list(range(299))
 #    'FS INT', 'SS INT', '15-FITC', '13-PE', '33-PC7', '2-APC', '7-APC-AF700',
  #   '34-ECD', '117-PC5.5', 'HLADR-PB', '45-CO'
# ] # List of channels to be used for training. Check spelling and consistency across samples. Adjust if needed.

# Define path to training data
training_data_path = './data/training'
# Get list of files in the data directory (only include ones ending with .csv)
training_files = sorted([fn for fn in os.listdir(training_data_path) if fn.endswith('.csv')])
# Load the training files into pandas dataframes
training_data_dfs_complete = [pd.read_csv(os.path.join(training_data_path, fn)) for fn in training_files]
training_set_df = training_data_dfs_complete[0].iloc[1:-1, trainchannels]
celltype_df = training_data_dfs_complete[0].iloc[1:-1, 300]  # Assuming the cell type is in column 300
celltype_df.replace(['AML-PB', 'MDS-PB', 'MPN-PB', 'NHL-PB', 'CLL-PB', 'ALL-PB', 
                     'Bacterial', 'Viral', 'post-reactive', 'CR-PB'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
# transform to numpy array for TSNE and apply arcsinh transformation
training_set = training_set_df.to_numpy()
training_set_t = np.arcsinh(training_set / 5)  # Apply arcsinh transformation with cofactor 5
celltype = celltype_df.to_numpy()

save_path = './results/TSNE-test'

tsne_model = TSNE(n_components=2, n_jobs=-1, initialization="pca", exaggeration=1.2, verbose=True)
# x_tsne = tsne_model.fit_transform(x_train)
x_tsne = tsne_model.fit(training_set_t)
# pickle.dump(x_tsne, open(os.path.join(save_path, 'tsne_embedding.pkl'), 'wb'))
# pickle.dump(tsne_model, open(os.path.join(save_path, 'tsne_model.pkl'), 'wb'))

fig = plt.figure(dpi=300)
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=10, alpha=0.5, c=celltype, cmap='tab10')

# Add colorbar with labels
cbar = plt.colorbar(scatter)
cell_type_labels = ['AML-PB', 'MDS-PB', 'MPN-PB', 'NHL-PB', 'CLL-PB', 'ALL-PB', 
                    'Bacterial', 'Viral', 'post-reactive', 'CR-PB']
cbar.set_ticks(range(len(cell_type_labels)))
cbar.set_ticklabels(cell_type_labels, fontsize=6)
cbar.set_label('Cell Type', rotation=270, labelpad=20)

fig.savefig(os.path.join(save_path, f'tsne_training_{date_time_str}.png'))
plt.close(fig)