# [Optimization Procedure] Neural Network Input Compression to 2-bit
## Table of Contents ##
* [Dataset](#dataset)
* [Dataset Preprocessing](#dataset-preprocessing)
* [Part 1: Optimizing Charge Thresholds](#part-1-optimizing-charge-thresholds)
    * [Path variables](#threshold-optimization-variables-optional-to-change)
    * [Threshold optimization variables](#threshold-optimization-variables-optional-to-change)
    * [Summary](#summary)
* [Part 2: Training on Optimized Charge Thresholds](#part-2-training-on-optimized-charge-thresholds)
    * [Threshold variables](#threshold-variables-only-if-skipping-part-1)
    * [Summary](#summary-1)
* [Part 3: Testing on dataset_2sc](#part-3-testing-on-dataset_2sc)

---
## Dataset ##
Our datasets are simulated using [TCAD Silvaco](https://silvaco.com/tcad/) for the sensor design and [PixelAV](https://cds.cern.ch/record/687440?ln=en) for the physics within the sensor.

The relevant datasets are labeled as `dataset_3sr_16x16_50x12P5_centeredIncidence_parquets` and `dataset_2s_16x16_50x12P5_centeredIncidence_parquets`. 
* `s` stands for Silvaco (This dataset was generating using TCAD Silvaco)
* `r` stands for Regression (This dataset was generated to improve regression trainings)
* `c` stands for Contained (This is a sub-dataset that is filtered to select contained clusters onl).
    * Contained clusters are defined to have < 50 units of charge summed from all edge pixels of the sensor array
* `16x16` represents the sensor array configuration (16x16 array of pixels)
* `50x12P5` represents the pixel dimenson (50um x 12.5um x 100um, pitch by thickness)

We are strictly concerned with the **contained cluster** subsets of these datasets. Our training set uses 80 files from dataset_3sr and our validation set uses a different 20 files from dataset_3sr. We test our final models using the same dataset_3sr validation set and a separate test on 100 files from dataset_2s.

You can find them using:
* **CERN EOS**
    * `/eos/project/s/smartpix-box/pixelAV_datasets/shuffled/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/train_contained`
    * `/eos/project/s/smartpix-box/pixelAV_datasets/shuffled/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/test_contained`
    * `/eos/project/s/smartpix-box/pixelAV_datasets/shuffled/largerWindowPreliminary/dataset_2s_16x16_50x12P5_centeredIncidence_parquets/test_contained`
* **CERNBox**:
    * [dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/train_contained](https://cernbox.cern.ch/files/spaces/eos/project/s/smartpix-box/pixelAV_datasets/shuffled/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/train_contained)
    * [dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/test_contained](https://cernbox.cern.ch/files/spaces/eos/project/s/smartpix-box/pixelAV_datasets/shuffled/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/test_contained)
    * [dataset_2s_16x16_50x12P5_centeredIncidence_parquets/test_contained](https://cernbox.cern.ch/files/spaces/eos/project/s/smartpix-box/pixelAV_datasets/shuffled/largerWindowPreliminary/dataset_2s_16x16_50x12P5_centeredIncidence_parquets/test_contained)

---
## Dataset Preprocessing ##
Before initiating any model training, we must first process the datasets from **parquet** format to **TFRecord** format. This is because TFRecords are TensorFlow's native binary data format and optimized for high-throughput model training. It makes the training faster and allows us to process very large datasets.

Fortunately, all of this is taken care of in the `two_bit_optimization.ipynb` notebook. **The notebook goes through the entire training/validation/testing pipeline,** while also handling the pre-processing step beforehand and any additional intermediate steps (such as saving model checkpoints).

In order to enable the dataset preprocessing, you must set `tfrecords_exist=False` and `select_contained=True` in the notebook. 
* `tfrecords_exist=False`: process the relevant parquet files, generate TFRecord copies, then save outputs to the specified directory. These TFRecords will then be loaded into training, validation, and test data-generators.
* `tfrecords_exist=True`: skip the TFRecord generation step. Load the existing TFRecords into training, validation, and test data-generators.
    * **Note: If you already generated the TFRecords for the relevant training, you can use this option**

In addition, you must create one directory for the dataset_3sr contained datasets and one directory for the dataset_2s contained dataset. The directory structure should look something like this:
```
smart_pixels_datasets/
├── dataset_3src_16x16_50x12P5_centeredIncidence_parquets/
│   ├── train_contained/
│   └── test_contained/
└── dataset_2sc_16x16_50x12P5_centeredIncidence_parquets/
    └── test_contained/
```
(As you can see, I renamed the parent directories to dataset_3sr**c**... and dataset_2s**c**... to be clear about using the contained datasets.)

The paths for these two directories will need to be added to the path variables in the notebook. So make sure to input the correct value into:
* `dataset_3src_dir=[YOUR dataset 3sr directory with its train_contained and test_contained datasets]`
* `dataset_2sc_dir=[YOUR dataset 2s directory with its test_contained dataset]`

Finally, make sure to input the desired model for training: `model_type=[MODEL TYPE]`

When the data preprocessing step is finished, each of your directories will contain a new `TFR_files` subdirectory, which holds the generated TFRecords in `.../TFR_files/2t` (2-timeslices, which is the default). These TFRecords will be named according to the arguments that you used to create it, for example: `TFR_train_contained` or `TFR_train_contained_slim_std_log`. 
* `slim`: 3 labels → x-midplane, y-midplane, cotBeta (for slim models ONLY)
* `std`: inputs are standardized (disabled by default)
* `log`: inputs are log-compressed (disabled by default)

This is an example of what the new directory structure might look like after the dataset preprocessing step is finished (running for a MAX or FULL model):
```
smart_pixels_datasets/
├── dataset_3src_16x16_50x12P5_centeredIncidence_parquets/
│   ├── train_contained/
│   ├── test_contained/
│   └── TFR_files/
│       └── 2t/
│           ├── TFR_train_contained/
│           └── TFR_test_contained/
└── dataset_2sc_16x16_50x12P5_centeredIncidence_parquets/
    ├── test_contained/
    └── TFR_files/
        └── 2t/
            └── TFR_test_contained/
```
If you are running the pipeline on a SLIM model, it will produce a `TFR_train_contained_slim` instead of `TFR_train_contained`, for example, and so on. This is taken care of internally.

## Part 1: Optimizing Charge Thresholds ##
The first step in the 2-bit input compression procedure is to identify the optimal charge thresholds. The model inputs consist of a 16×16 array of two-channel charge values. The objective is to partition these values into four discrete bins, corresponding to the 2-bit encodings (00, 01, 10, 11), such that model performance (measured by the loss) is optimized. This is equivalent to determining three bin boundaries, or physically, three charge thresholds.

The relevant variables to edit are:

#### Path variables
* `weights_directory`: the directory path where you want to save all of the model checkpoints (hdf5 format)
* `performance_directory`: the directory path where you want to save the results of testing your model on the test sets (parquet format)

#### Threshold optimization variables (OPTIONAL TO CHANGE)
* `initial_thresholds=[247.8, 668.4, 1662.9]`: these are the starting values for the 3 charge thresholds that were determined from the optimal values trained on a transformer. You can leave this as is or test out your own starting values.
* `threshold_offset=80.0`: this is the standard offset that we have used for all of our trainings, representing 1 standard deviation of charge produced by sensor noise. You can leave this as is or try training without an offset to see if it can potentially improve performance.

You also have the option to skip Part 1 of this notebook by setting the flag `skip_part_1=True`.

#### Summary
Part 1 of the optimization procedure will generate and load the TFRecords for dataset_3src into training and validation data-generators, add Gaussian noise (mean=0e, stdev=80e), create the desired model with the soft quantize layer, then train on it for 1000 epochs. After each epoch, its model checkpoint will be saved to `weights_directory` with naming convention `weights-[TIMESLICES]t-[MODEL TYPE]-soft_quantize_layer-[FINGERPRINT ID]-checkpoints`. When the final epoch is finished, the best model checkpoint (epoch with lowest validation loss) will be automatically selected and its desired charge thresholds will be extracted from its soft quantize layer. Afterwards, the training and validation data-generators are deleted to free up storage.

## Part 2: Training on Optimized Charge Thresholds ##
In this stage of the pipeline, all the required arguments in the notebook are already set (if you followed the above instructions). You do not have to change anything for this part to work. 

#### Threshold variables (ONLY IF SKIPPING PART 1)
* `skip_part_1=True`: skip all steps in Part 1 (use this if you don't need to determine charge thresholds again and want to use your own
* `thresholds=[value1, value2, value3]`: these are your custom thresholds that you want to use instead of using Part 1's thresholds
* `levels=np.array([0.0, 1.0, 2.0, 3.0] dtype=np.float32)`: You don't need to change this -- these are just the output values of the 4 bins.

#### Summary
Part 2 of the optimization procedure will load the TFRecords for dataset_3src into training and validation data-generators using the thresholds from Part 1. Noise will not be added in this step. The desired model will be created without the soft quantize layer and it will be trained on for 1000 epochs. After each epoch, its model checkpoint will be saved to `weights_directory` with naming convention `weights-[TIMESLICES]t-[MODEL TYPE]-2bit_optimized-[FINGERPRINT ID]-checkpoints`. When the final epoch is finished, the best model checkpoint (epoch with lowest validation loss) will be automatically selected and tested on by the dataset_3src test set. The resulting file will be saved to `performance_directory`.

## Part 3: Testing on dataset_2sc



