# FLAG-X tutorial

## Resources
- Package: [GitHub](https://github.com/bionetslab/FLAG-X/tree/main), [Bioconda](https://anaconda.org/channels/bioconda/packages/flagx/overview)
- Documentation: [Read the Docs](https://flag-x.readthedocs.io/en/latest/)

## General
FLAG-X is used for the automatic classification of cell populations in flow cytometry data. The scripts provided here can be used to create an effective workflow.

The files used for training and inference must contain the same channels and must have been acquired under roughly the same measurement conditions.

## Recommende Workflow

### Data Verification and Preparation

**check_input** helps verify which channels are present in the files and whether they match across all files (currently only for CSV).

**csv_format** Assigns the same channel names to all files. If necessary, deletes individual channels that are not present in all files.

**config_xxx.yml** files should be used to define parameters for one training-inference task. Depending on the task, not all channels of the fcs files are used for training-inference.

### Training

For the training, compile datasets that contain all expected cell types at least once in sufficient cell numbers. Not every population needs to contain every cell type.

**training_from_fcs** Script runs with fcs and csv. Concatenates the datasets, performs dimensionality reduction with SOM and TSNE, and performs automatic clustering using PARC. The raw data is exported as an FCS file along with the metadata thus calculated. The FCS file can then be gated using flow cytometry software (we use Kaluza for the ease of color coding), with the calculated data (SOM, TSNE, PARC) facilitating the definition of the cell populations. 

Export data from cell populations of interest population-wise and concatenate data into one large csv with a column indicating the cell populations as 1, 2, 3...

**training_labelled_csv** generates a SOM model and an MLP model on the compound csv containing the population annotations. An FCS is also provided for control purposes.

### Inference

**fcs_inference** calculates the assignment of cells in new files to the defined populations based on the models generated above. The population statistics (cell numbers and (optional) fluorescence intesities) are exported as a CSV table with one row for each file. The MLP model is routinely used for this purpose. An FCS file, which can be used to verify the population assignments via color labeling, is also exported.



