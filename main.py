

def workflow_pipeline_unsupervised():

    import os

    from flagx import GatingPipeline

    save_dir = './results/workflow_pipeline_unsupervised'
    os.makedirs(save_dir, exist_ok=True)

    channels = [
        'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
        'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
        'FL10 INT_CD45-KO'
    ]

    cutoffs = {
        'FS INT': 1000, 'SS INT': 500,
        'FL1 INT_CD14-FITC': 300, 'FL2 INT_CD19-PE': 300, 'FL3 INT_CD13-ECD': 300, 'FL4 INT_CD33-PC5.5': 300,
        'FL5 INT_CD34-PC7': 400, 'FL6 INT_CD117-APC': 400, 'FL7 INT_CD7-APC700': 400, 'FL8 INT_CD16-APC750': 400,
        'FL9 INT_HLA-PB': 100, 'FL10 INT_CD45-KO': 100,
    }

    preprocessing_kwargs = {'flavour': 'log10_w_custom_cutoffs', 'flavour_kwargs': {'cutoffs': cutoffs}}

    som_kwargs = {
        'som_topology': 'planar',
        'som_grid_type': 'rectangular',
        'som_dimensions': (6, 6),
        'neighborhood': 'gaussian',
        'gaussian_neighborhood_sigma': 0.1,
        'initialization': 'pca',
        'n_epochs': 10,
        'verbosity': 3,
    }

    # Initialize the pipeline
    gp = GatingPipeline(
        train_data_file_path='./data/training',
        train_data_file_names=None,
        train_data_file_type=None,
        save_path='./results/workflow_pipeline_unsupervised/gp_output',
        channels=channels,
        label_key=None,  # No labels
        compensate=False,
        channel_names_alignment_kwargs=None,
        relabel_data_kwargs=None,
        preprocessing_kwargs=preprocessing_kwargs,
        downsampling_kwargs={'target_num_events': 100},
        gating_method='som',
        gating_method_kwargs=som_kwargs,
    )

    # Train the underlying ML model
    gp.train()

    # Annotate and export data
    gp.inference(
        data_file_path='./data/testing',
        data_file_names=None,
        sample_wise=False,  # Export all into one file
        gate=False,  # Cannot gate since SOM was trained with unlabeled data
        dim_red_methods=('som', 'umap'),
        dim_red_method_kwargs=(None, {'n_jobs': -1}),
        save_path=save_dir,
        save_filename='annotated_data.fcs',
        val_range=(0, 2**20),
        keep_unscaled=False,
    )

    # Save the pipeline
    gp.save(filepath=save_dir)


def workflow_step_wise_unsupervised_som_training():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from flagx.io import FlowDataManager
    from flagx.gating import SomClassifier

    # --- Define path where results are saved to
    save_path = './results/workflow_step_wise_unsupervised_som_training'
    save_path_data_handling = './results/workflow_step_wise_unsupervised_som_training/data_handling'
    os.makedirs(save_path_data_handling, exist_ok=True)

    # --- Specify where training data is saved and specify the corresponding filenames
    # Define path to training data
    training_data_path = './data/flagx_tutorial_data/training'

    # Get list of files in the data directory (only include ones ending with .csv)
    training_files = sorted([fn for fn in os.listdir(training_data_path) if fn.endswith('.csv')])

    # --- Before data processing with the FlowDataManager, check whether each file contains the same number of channels
    # Load the training files into pandas dataframes
    training_data_dfs = [pd.read_csv(os.path.join(training_data_path, fn)) for fn in training_files]

    # For each file print the number of channels
    for fn, data_df in zip(training_files, training_data_dfs):
        print(f'# --- {fn}, number of channels: {data_df.shape[1]}')

    print('\n')

    # As an additional check, also print the channel names
    for fn, data_df in zip(training_files, training_data_dfs):
        print(f'# --- {fn}:\n{data_df.columns.to_list()}')

    print('\n')

    # RESULT:
    #  - Number of channels is consistent across samples
    #  - If one sample had 14 instead of 13 channels (e.g. FS Peak),
    #    this channel should be removed to match the other samples.


    # --- Data loading and processing
    # Initialize the data manager
    fdm = FlowDataManager(
        data_file_names=training_files,
        data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'training_files' list
        data_file_path=training_data_path,
        save_path=save_path_data_handling,
        verbosity=1
    )

    # Load data into memory
    # Data samples are now stored not in a Pandas DataFrames, but in a list of AnnData object.
    # This list is an attribute of the FlowDataManager class and can be accessed via FlowDataManager.anndata_list_.
    # AnnData is a Python class similar to Pandas DataFrames but with more options and functions for data annotation.
    # see: https://anndata.readthedocs.io/en/stable/
    fdm.load_data_files_to_anndata()
    # Print the loaded AnnData objects containing the loaded training data
    for adata in fdm.anndata_list_:
        print(f'# --- {adata.uns['filename']}:\n{adata}')

    # --- Check the number of events per sample
    # Create a dataframe with the columns sample and n_events, this df is an attribute of the FDM instance
    # and can be accessed via FDM.sample_sizes_. Since a filename is passed as well, the df is also saved in the
    # directory specified via 'save_path_data_handling'.
    fdm.check_sample_sizes(filename_sample_sizes_df='sample_sizes.csv')

    # Use a built-in plotting function to visualize the number of events per sample.
    # Resulting plot also saved to 'save_path_data_handling'.
    fig, ax = plt.subplots(dpi=300)
    fdm.plot_sample_size_df(sample_size_df=fdm.sample_sizes_, ax=ax)
    fig.savefig(os.path.join(save_path_data_handling, 'sample_sizes.png'))
    plt.close(fig)

    # --- Apply preprocessing transformation to each sample
    # Example 1: Apply arcsinh with cofactor 150,
    # Example 2: Apply log transformation with custom cutoffs
    # In both cases, store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
    example_1 = True
    if example_1:
        preprocessing_kwargs = {'cofactor': 150}
        fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
    else:
        # Define python dictionary mapping channel names to cutoffs (arbitrarily chosen here, adjust if needed)
        channel_name_to_cutoff = {
            'FS INT': 1000, 'SS INT': 800,
            '15-FITC': 300, '13-PE': 300, '34-ECD': 300, '117-PC5.5': 300, '33-PC7': 300,
            '2-APC': 200, '7-APC-AF700': 200, '-APC-AF750': 200, 'HLADR-PB': 200, '45-CO': 200,
        }
        preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
        fdm.sample_wise_preprocessing(
            flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
        )

    # --- Downsample each sample to a target number of events
    # Set target_num_events to 1000 for fast model training in this example
    fdm.sample_wise_downsampling(data_set='all', target_num_events=1000)

    # --- Extract concatenated data matrix for model training
    # Define channels to be used for model training
    channels = [
        'FS INT', 'SS INT',
        '15-FITC', '13-PE', '34-ECD', '117-PC5.5', '33-PC7', '2-APC', '7-APC-AF700', '-APC-AF750', 'HLADR-PB', '45-CO'
    ]
    # Extract the processed data matrices from the AnnData objects
    data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
    # Concatenate
    x_train = np.concatenate(data_matrices, axis=0)
    # Shuffle
    idx_shuffle = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_shuffle]

    # --- SOM training
    # Instantiate the SOMClassifier, set hyperparameters
    som_clf = SomClassifier(
        som_topology='planar',
        som_grid_type='rectangular',
        som_dimensions=(6, 3),  # (25, 25)
        neighborhood='gaussian',
        gaussian_neighborhood_sigma=0.1,
        initialization='pca',
        n_epochs=100,  # 1000,
        radius_0=-0.25,
        radius_n=0.1,
        radius_cooling='exponential',
        learning_rate_0=0.1,
        learning_rate_n=0.001,
        learning_rate_decay='exponential',
        unlabeled_label=-999,
        verbosity=2
    )

    # Model fitting. Since no labels are available only the SOM component is trained in an unsupervised fashion.
    # Still, a dummy label vector must be passed containing the label chosen to indicate unlabeled events
    y_dummy = np.ones(x_train.shape[0]) * -999
    som_clf.fit(X=x_train, y=y_dummy)

    # Save the trained model
    som_clf.save(filename='som_classifier.pkl', filepath=save_path)


def workflow_step_wise_unsupervised_inference():
    import os
    import numpy as np
    import pandas as pd

    from flagx.io import FlowDataManager, export_to_fcs
    from flagx.gating import SomClassifier
    from flagx.dimred import TSNE, UMAP

    # --- Define path where results are saved to
    save_path = './results/workflow_step_wise_unsupervised_inference'
    os.makedirs(save_path, exist_ok=True)


    # --- Specify where test data is saved and specify the corresponding filenames
    # Define path to training data
    test_data_path = './data/flagx_tutorial_data/testing'

    # Get list of files in the data directory (only include ones ending with .csv)
    test_files = sorted([fn for fn in os.listdir(test_data_path) if fn.endswith('.csv')])

    # --- Before data processing with the FlowDataManager, check whether each file contains the same number of channels
    # Load the training files into pandas dataframes
    test_data_dfs = [pd.read_csv(os.path.join(test_data_path, fn)) for fn in test_files]

    # For each file print the number of channels
    for fn, data_df in zip(test_files, test_data_dfs):
        print(f'# --- {fn}, number of channels: {data_df.shape[1]}')

    print('\n')

    # As an additional check, also print the channel names
    for fn, data_df in zip(test_files, test_data_dfs):
        print(f'# --- {fn}:\n{data_df.columns.to_list()}')

    print('\n')

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # The test data files contain an error:
    # Channel '-APC-AF750' is named '#NAME?' (probably an artifact from manual renaming of the channels).
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Fix the naming error and save the corrected files into a newly created directory
    # Name the column correctly
    for data_df in test_data_dfs:
        data_df.rename(columns={'#NAME?': '-APC-AF750'}, inplace=True)
        # print(data_df.columns.to_list())

    # Create a new dir, save, and reset test_data_path
    test_data_path_correct = './data/flagx_tutorial_data/testing_corrected'
    os.makedirs(test_data_path_correct, exist_ok=True)
    for fn, data_df in zip(test_files, test_data_dfs):
        data_df.to_csv(os.path.join(test_data_path_correct, fn), index=False)


    # --- Data loading and processing; NOTE: data should be processed the same way as for model training
    # Initialize the data manager
    fdm = FlowDataManager(
        data_file_names=test_files,
        data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'test_files' list
        data_file_path=test_data_path_correct,
        verbosity=1
    )

    # Load data into memory
    fdm.load_data_files_to_anndata()

    # --- Apply preprocessing transformation to each sample, same as for training!
    # Example 1: Apply arcsinh with cofactor 150,
    # Example 2: Apply log transformation with custom cutoffs
    # In both cases, store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
    example_1 = True
    if example_1:
        preprocessing_kwargs = {'cofactor': 150}
        fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)
    else:
        # Define python dictionary mapping channel names to cutoffs (arbitrarily chosen here, adjust if needed)
        channel_name_to_cutoff = {
            'FS INT': 1000, 'SS INT': 800,
            '15-FITC': 300, '13-PE': 300, '34-ECD': 300, '117-PC5.5': 300, '33-PC7': 300,
            '2-APC': 200, '7-APC-AF700': 200, '-APC-AF750': 200, 'HLADR-PB': 200, '45-CO': 200,
        }
        preprocessing_kwargs = {'cutoffs': channel_name_to_cutoff}
        fdm.sample_wise_preprocessing(
            flavour='log10_w_custom_cutoffs', save_raw_to_layer='no_trafo', **preprocessing_kwargs
        )

    # NOTE: typically no downsampling at inference time, do so for faster UMAP and t-SNE computation
    # --- Downsample each sample to a target number of events
    fdm.sample_wise_downsampling(data_set='all', target_num_events=100)

    # Define channels that were used for model training
    channels = [
        'FS INT', 'SS INT',
        '15-FITC', '13-PE', '34-ECD', '117-PC5.5', '33-PC7', '2-APC', '7-APC-AF700', '-APC-AF750', 'HLADR-PB', '45-CO'
    ]
    # Extract the processed data matrices from the AnnData objects
    data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
    # Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
    num_events = [x.shape[0] for x in data_matrices]
    starting_indices = np.cumsum([0, ] + num_events)
    # Concatenate
    x_test = np.concatenate(data_matrices, axis=0)

    # --- Compute dimensionality reductions on concatenated test samples
    # --- SOM
    # Load the previously trained SOM model
    som_clf = SomClassifier.load(
        filename='som_classifier.pkl',
        filepath='./results/workflow_step_wise_unsupervised_som_training'
    )
    _, x_som, _, _ = som_clf.transform(x_test)

    # --- UMAP
    umap_model = UMAP(n_components=2, n_jobs=-1)
    x_umap = umap_model.fit_transform(x_test)

    # --- t-SNE
    tsne_model = TSNE(n_components=2, n_jobs=-1)
    x_tsne = tsne_model.fit_transform(x_test)

    # Change back into sample-wise format (input format required by export function)
    x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
    x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
    x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
    x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

    # Export to FCS
    export_to_fcs(
        data_list=fdm.anndata_list_,  # Export the test samples
        layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
        sample_wise=False,  # Export one FCS in which the test samples are concatenated
        add_columns=[
            x_soms_1, x_soms_2,
            x_umaps_1, x_umaps_2,
            x_tsnes_1, x_tsnes_2
        ],  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
        add_columns_names=['SOM_1', 'SOM_2', 'UMAP_1', 'UMAP_2', 'TSNE_1', 'TSNE_2'],  # Add names for added columns
        scale_columns=['SOM_1', 'SOM_2', 'UMAP_1', 'UMAP_2', 'TSNE_1', 'TSNE_2'],  # Select added columns for scaling
        val_range=(0, 2**20),  # Range to which selected columns are scaled to
        save_path=save_path,
        save_filenames='annotated_test_data.fcs'
    )


def example_concatenating_csv_files():

    import os
    import numpy as np
    import pandas as pd

    # --- Create example CSV files ---

    # Create directory to save example files into
    data_path = './data/example_csv'
    os.makedirs(data_path, exist_ok=True)

    # 5 gated populations
    population_names = ['others', 'blasts', 'laip1', 'laip2']

    # 3 channels + sample ID
    num_channels = 3
    channel_names = [f'C_{i}' for i in range(num_channels)] + ['sample_id']

    # Iterate over populations and create example data
    for population_name in population_names:

        # Generate random data matrix
        n_events = 10
        x_data = np.random.normal(size=(n_events, num_channels), loc=0, scale=1).round(decimals=4)

        # Generate random sample IDs between 1 and 6
        x_sample_id = np.random.randint(low=1, high=6, size=n_events)

        # Add sample ID as column to data matrix
        x_sample_id = x_sample_id.reshape(-1, 1)
        x_data_concat = np.concatenate((x_data, x_sample_id), axis=1)

        # Create DataFrame with channel names as column names
        data_df = pd.DataFrame(x_data_concat, columns=channel_names)

        # Save
        data_df.to_csv(os.path.join(data_path, f'{population_name}.csv'), index=False)


    # --- Concatenate CSVs into one file and add population labels ---
    filenames = ['others.csv', 'blasts.csv', 'laip1.csv', 'laip2.csv']
    data_dfs = []
    meta_data = []
    for i, filename in enumerate(filenames):

        # Load the CSV file
        data_df = pd.read_csv(os.path.join(data_path, filename))

        # Add a columns with an integer label encoding the population
        data_df['label'] = i

        # Add dataframe to list
        data_dfs.append(data_df)

        # Save information which population is encoded by which integer
        meta_data.append({
            'filename': filename,
            'integer_label': i
        })

    # Concatenate the DataFrames and save again
    data_df_concat = pd.concat(data_dfs, axis=0, ignore_index=True)
    data_df_concat.to_csv(os.path.join(data_path, 'data_concatenated_with_population_labels.csv'), index=False)

    # Generate and save DataFrame with the label information
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.to_csv(os.path.join(data_path, 'meta_data.csv'), index=False)


def workflow_step_wise_supervised_training():
    import os
    import torch
    import numpy as np
    import pandas as pd

    from flagx.io import FlowDataManager
    from flagx.gating import SomClassifier, MLPClassifier

    # --- Define path where results are saved to
    save_path = './results/workflow_step_wise_supervised_training'
    save_path_data_handling = './results/workflow_step_wise_supervised_training/data_handling'
    os.makedirs(save_path_data_handling, exist_ok=True)

    # --- Generate some artificial training data for this example
    # Define path to training data
    training_data_path = './data/supervised_example'
    os.makedirs(training_data_path, exist_ok=True)

    num_channels = 5
    num_events = 100
    channel_names = [f'C_{i}' for i in range(num_channels)] + ['label']
    for i in range(6):
        # Generate random data matrix
        x_data = np.random.normal(size=(num_events, num_channels), loc=i, scale=0.01)
        # Generate random labels between i and i + 2
        y_label = np.random.randint(low=i, high=i + 2, size=num_events)
        # Add labels as column to data matrix
        y_label = y_label.reshape(-1, 1)
        x_data_concat = np.concatenate((x_data, y_label), axis=1)
        # Create DataFrame with channel names as column names
        data_df = pd.DataFrame(x_data_concat, columns=channel_names)
        # Save
        data_df.to_csv(os.path.join(training_data_path, f'sample_{i}.csv'), index=False)

    # Get list of files in the data directory (only include ones ending with .csv)
    training_files = sorted([fn for fn in os.listdir(training_data_path) if fn.endswith('.csv')])

    # --- Data loading and processing
    # Initialize the data manager
    fdm = FlowDataManager(
        data_file_names=training_files,
        data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'training_files' list
        data_file_path=training_data_path,
        save_path=save_path_data_handling,
        verbosity=1
    )

    # Load data into memory
    # Data samples are now stored not in a Pandas DataFrames, but in a list of AnnData object.
    # This list is an attribute of the FlowDataManager class and can be accessed via FlowDataManager.anndata_list_.
    # AnnData is a Python class similar to Pandas DataFrames but with more options and functions for data annotation.
    # see: https://anndata.readthedocs.io/en/stable/
    fdm.load_data_files_to_anndata()

    # --- Apply preprocessing transformation to each sample
    # Apply arcsinh with cofactor 150.
    # Store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
    preprocessing_kwargs = {'cofactor': 150}
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)

    # --- Downsample each sample to a target number of events
    # Set target_num_events to 100 for fast model training in this example
    fdm.sample_wise_downsampling(data_set='all', target_num_events=50)

    # --- Extract concatenated data matrix for model training
    # Define channels to be used for model training
    channels = [f'C_{i}' for i in range(num_channels)]

    # Use dataloader with batchsize -1 (= all data) to extract the transformed data matrix and label vector
    # Note that the data matrix from which we retrieve the labels is the original non-transformed data (label_layer_key='no_trafo').
    # Otherwise, we would get the arcsinh-transformed labels
    data_loader = fdm.get_data_loader(
        data_set='all',
        channels=channels,
        label_key='label',
        label_layer_key='no_trafo',
        batch_size=-1,
    )
    x_train, y_train = next(iter(data_loader))

    TRAIN_SOM = True
    if TRAIN_SOM:
        # Instantiate the SOMClassifier, set hyperparameters
        som_clf = SomClassifier(
            som_topology='planar',
            som_grid_type='rectangular',
            som_dimensions=(3, 3),  # (25, 25)
            neighborhood='gaussian',
            gaussian_neighborhood_sigma=0.1,
            initialization='pca',
            n_epochs=100,  # 1000,
            radius_0=-0.25,
            radius_n=0.1,
            radius_cooling='exponential',
            learning_rate_0=0.1,
            learning_rate_n=0.001,
            learning_rate_decay='exponential',
            unlabeled_label=-999,
            verbosity=2
        )
        # Model fitting
        som_clf.fit(X=x_train, y=y_train)
        # Save the trained model
        som_clf.save(filename='som_classifier.pkl', filepath=save_path)

    else:
        # Instantiate the MLP, set hyperparameters
        mlp_clf = MLPClassifier(
            layer_sizes=(128, 64, 32),
            n_epochs=100,
            data_loader_params={'batch_size': 128, 'shuffle': True, 'num_workers': 1},
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbosity=2
        )
        # Model fitting
        mlp_clf.fit(X=x_train, y=y_train)
        # Save the trained model
        mlp_clf.save(filename='mlp_classifier.pkl', filepath=save_path)


def workflow_step_wise_supervised_inference():
    import os
    import numpy as np

    from flagx.io import FlowDataManager, export_to_fcs
    from flagx.gating import SomClassifier, MLPClassifier
    from flagx.dimred import UMAP, TSNE

    # --- Define path where results are saved to
    save_path = './results/workflow_step_wise_supervised_inference'
    os.makedirs(save_path, exist_ok=True)

    # Define path to training data
    inference_data_path = './data/supervised_example'

    # Get list of files in the data directory (only include ones ending with .csv)
    inference_files = sorted([fn for fn in os.listdir(inference_data_path) if fn.endswith('.csv')])

    # --- Data loading and processing
    # Initialize the data manager
    fdm = FlowDataManager(
        data_file_names=inference_files,
        data_file_type=None,  # Is inferred from the filename ending of the 1st file in the 'training_files' list
        data_file_path=inference_data_path,
        verbosity=1
    )

    # Load data into memory
    fdm.load_data_files_to_anndata()

    # --- Apply preprocessing transformation to each sample, same as for training data
    # Apply arcsinh with cofactor 150.
    # Store non-transformed data in a separate layer of the AnnData object that we call 'no_trafo'.
    preprocessing_kwargs = {'cofactor': 150}
    fdm.sample_wise_preprocessing(flavour='arcsinh', save_raw_to_layer='no_trafo', **preprocessing_kwargs)

    # Define channels that were used for model training
    channels = [f'C_{i}' for i in range(5)]

    # Extract the processed data matrices from the AnnData objects
    data_matrices = [adata[:, channels].X for adata in fdm.anndata_list_]
    # Get number of events per test sample and compute indices at which samples start in the concatenated data matrix
    num_events = [x.shape[0] for x in data_matrices]
    starting_indices = np.cumsum([0, ] + num_events)
    # Concatenate
    x_test = np.concatenate(data_matrices, axis=0)

    # Load the previously trained models
    som_clf = SomClassifier.load(
        filename='som_classifier.pkl',
        filepath='./results/workflow_step_wise_supervised_training'
    )

    mlp_clf = MLPClassifier.load(
        filename='mlp_classifier.pkl',
        filepath='./results/workflow_step_wise_supervised_training'
    )

    # Make prediction for the test data
    y_pred_som = som_clf.predict(x_test)
    y_pred_mlp = mlp_clf.predict(x_test)

    # Change predictions back into sample-wise format (input format required by export function)
    y_pred_som = [y_pred_som[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]
    y_pred_mlp = [y_pred_mlp[starting_indices[i]: starting_indices[i + 1]] for i in range(len(num_events))]


    add_columns = [y_pred_som, y_pred_mlp]
    add_columns_names = ['pred_som', 'pred_mlp']

    # If dimensionality reductions should be computed as well, set: compute_dim_red = True
    compute_dim_red = True
    if compute_dim_red:

        # --- SOM
        _, x_som, _, _ = som_clf.transform(x_test)

        # --- UMAP
        umap_model = UMAP(n_components=2, n_jobs=-1)
        x_umap = umap_model.fit_transform(x_test)

        # --- t-SNE
        tsne_model = TSNE(n_components=2, n_jobs=-1)
        x_tsne = tsne_model.fit_transform(x_test)

        # Change back into sample-wise format (input format required by export function)
        x_soms_1 = [x_som[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
        x_soms_2 = [x_som[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
        x_umaps_1 = [x_umap[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
        x_umaps_2 = [x_umap[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]
        x_tsnes_1 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 0] for i in range(len(num_events))]
        x_tsnes_2 = [x_tsne[starting_indices[i]: starting_indices[i + 1], 1] for i in range(len(num_events))]

        add_columns += [
            x_soms_1, x_soms_2,
            x_umaps_1, x_umaps_2,
            x_tsnes_1, x_tsnes_2
        ]
        add_columns_names += ['SOM_1', 'SOM_2', 'UMAP_1', 'UMAP_2', 'TSNE_1', 'TSNE_2']

    # Export to FCS
    export_to_fcs(
        data_list=fdm.anndata_list_,  # Export the test samples
        layer_key='no_trafo',  # We want to export non-transformed data => choose the 'no_trafo' layer
        sample_wise=False,  # Export one FCS in which the test samples are concatenated
        add_columns=add_columns,  # Add columns corresponding to the 1st and 2nd dimension of the dimensionality reductions into 2D
        add_columns_names=add_columns_names,  # Add names for added columns
        scale_columns=add_columns_names,  # Select added columns for scaling (all that were added to the file)
        val_range=(0, 2 ** 20),  # Range to which selected columns are scaled to
        save_path=save_path,
        save_filenames='annotated_test_data.fcs'
    )




if __name__ == '__main__':

    # workflow_pipeline_unsupervised()

    # workflow_step_wise_unsupervised_som_training()

    # workflow_step_wise_unsupervised_inference()

    # example_concatenating_csv_files()

    # workflow_step_wise_supervised_training()

    workflow_step_wise_supervised_inference()

    print('done')




