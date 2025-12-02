## How can I determine the optimal 2-bit charge bin configuration for the ASIC inputs?
Our strategy to determine the most optimal charge bins relies on attaching a primary "Soft Quantize Layer" to our various machine learning models. This layer initializes four smooth sigmoids that gradually become steeper as the training progresses, eventually becoming steep enough to separate the inputs into different "bins". The mean of each sigmoid is a trainable parameter and the goal is for the model to learn best sigmoid means (representing the charge thresholds/bin boundaries), while simultaneously predicting outputs that minimize the loss.

## What is the procedure that I have to follow in order to recreate this optimization process?
The training comes in 2 main parts. 

**Part 1**: Train the model with the soft quantize layer on a dataset injected with gaussian noise (mean = 0e, sigma = 80e). This dataset should not have any standardization, log compression, or digitization/quantization.

**Part 2**: Train the model (omit the soft quantize layer) on the same dataset but digitized to 2-bits according to the 3 charge thresholds obtained from part-1. This dataset should not have any standardization or log compression, but will be digitized to the values of 0.0, 1.0, 2.0 and 3.0.


In the notebook `two_bit_optimization.ipynb`, we have the following arguments:
* `dataset_dir`: Where your dataset is located. It should have `train`, `test`, `train_contained`, `test_contained` like on CERNbox
* `weights_dir`: The notebook will save the part-1 training checkpoint directory and the part-2 training checkpoint directory here.
* `performance_dir`: The notebook will save the final parquet file here, which will contain the performance variables (residuals_x, sigmacotB, etc.) of the best model tested on the test set.
* `model_type`: The model that you are training with. For non-quantized models, we have Conv2D_Max, Conv2D_Full, Conv2D_Slim, Conv1D_Full, Conv1D_Slim, Mlp_Full, and Mlp_Slim. For quantized models, you just need to add a "Q" to the front (ex: QConv2D_Full or QMlp_Slim).
* `tfrecords_exist`: Set to `False` if you previously have not generated the TFRecords with this notebook. If you are re-running the training pipeline and using the same TFRecords or you are training with a similar model (Full and Max both use the same TFRecords of 4 labels while Slim only uses 3 labels for its TFRecords), then you can set to `True` to save some time. Note: if you are running 2 or more copies of the notebook and generating the TFRrecords from scratch in parallel, it will overwrite one another. Therefore, make sure you finish generating the TFRecords for the first notebook that you set the flag to `False`, then set to `True` for the next notebooks you run in parallel.

The other arguments shouldn't really be changed unless you know what you are doing. This is because we are setting these parameters as fixed for the consistency between the different model trainings and I made them argument just to give the notebook more flexibility (if needed).

---
## How can I extract the charge thresholds from part 1? 
`two_bit_optimization.ipynb` automatically extracts and prints the values of the charge thresholds. However, most of us want to just click "run all cells" and exit the notebook to let it run in the background to do its thing. However, you can use `get_best_thresholds()` located in `two_bit_optimization_helpers/train.py` to print out these thresholds. Just start up a fresh notebook or script, feed it the path of the part-1 checkpoints directory, model type, threshold offset, etc. and it will return the thresholds and levels. An example is shown in `training_tracker.ipynb` on how to import the function to the notebook (first cell) and use it (last cell).

