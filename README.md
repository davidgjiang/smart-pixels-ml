# [Optimization Procedure] Neural Network Input Compression to 2-bit
## Table of Contents ##
* [Dataset](#dataset)
* [Dataset Processing](#dataset-processing)
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
## Dataset Processing ##
Before initiating any model training, we must first process the datasets from **parquet** format to **TFRecord** format. This is because TFRecords are TensorFlow's native binary data format and optimized for high-throughput model training. It makes the training faster and allows us to process very large datasets.






