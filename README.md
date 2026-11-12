# FLAG-X UseCases

## Resources
- Flag-X Package: [GitHub](https://github.com/bionetslab/FLAG-X/tree/main), [Bioconda](https://anaconda.org/channels/bioconda/packages/flagx/overview)
- Documentation: [Read the Docs](https://flag-x.readthedocs.io/en/latest/)

## General
FLAG-X is used for the automatic classification of cell populations in flow cytometry data. The scripts provided here show examples to create an effective workflow using this tool in combination with a conventional flow cytometry software (the authors use Kaluza).

The files used for training and inference must contain the same channels and must have been acquired under roughly the same measurement conditions.

## Recommended Workflow

### Data Verification and Preparation

**check_input** helps verify which channels are present in the files and whether they match across all files.

**csv_format** Assigns the same channel names to all files. If necessary, deletes individual channels that are not present in all files.

**config_xxx.yml** files should be used to define parameters for one training-inference task. Depending on the task, not all channels of the fcs files are used for training and inference.

**fcs_preselection** is a modification of fcs_inference described below. A simple training and inference loop applying gating with only few markers can be used to enrich cells of interest by differential downsampling in large fcs files.

### Training

For the training, compile datasets that contain all expected cell types ("populations") at least once in sufficient cell numbers. Not every sample needs to contain every cell type.

For the suggested workflow, two training runs are performed, one unsupervised run to concatenate files and perform dimensionality reduction to support gating, one unsupervised run to train the models for subsequent inference of new files.

**training_from_fcs** Script runs with fcs, lmd and csv. Applies downsampling if needed, concatenates the datasets, performs dimensionality reduction with SOM and TSNE, and performs automatic clustering using PARC. The raw data is exported as an FCS file along with the metadata (SOM map, TSNE map, PARC labels) thus calculated. 

After loading and compensation, for the purpose of harmonization, `training_from_fcs` removes channels named in `delete_channels` from each file (names not present in a file are ignored). When `harmonize_channels` is true, the remaining channels are renamed positionally using `all_channels`: every file must have the same number of remaining channels, in the same physical order, and `all_channels` must contain unique names. 

The FCS file contining concatenated data from several samples can then be gated using flow cytometry software (we use Kaluza for the ease of color coding), with the calculated data (SOM, TSNE, PARC) facilitating the definition of the cell populations. Of course definition of cell populations in raw training files one by one instead of concatenated files is also possible.

Export data from cell populations of interest population-wise (one population per exported csv). Alternatively, if your software offers this option, export data with a column indicating the cell populations as 1, 2, 3...

**concatenate_csv** concatenates population-specific csv into one large csv with a column indicating the cell populations. Naming conventions are given in the script. Optionally, a heat map of marker expression in the different populations ist caluclated.

**training_labelled_csv** generates a SOM model and an MLP model on the compound csv containing the population annotations. An FCS is also provided for control purposes.

### Inference

**fcs_inference** calculates the assignment of cells in new files to the training populations based on the models generated above. The population statistics (cell numbers and (optional) fluorescence intensities) are exported as a CSV table with one row for each file. The MLP model is routinely used for this purpose. An FCS file, which can be used to verify the population assignments via color labeling, is also exported.

Inference applies the same `delete_channels` and optional positional `harmonize_channels` settings as `training_from_fcs`, after compensation and before preprocessing. Use the same channel order and names as during training, including for `trainchannels`, `calcchannels` and transformation cutoffs. The inference summary records deleted channels once across all batches and, when harmonizing, the original and harmonized channel names.

Inference randomly downsamples each input file to at most `size_per_sample` events before batching, compensation and prediction. Batching uses the retained event counts; the file that takes a batch over `max_rows_per_batch` remains in that batch. Population counts, medians, and exported FCS data describe the retained events, not the full input file. Files smaller than the limit are kept in full.
