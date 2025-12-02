## How can I determine the optimal 2-bit charge bin configuration for the ASIC inputs?
Our approach identifies optimal charge-bin boundaries by attaching a Soft Quantize Layer to each machine-learning model. This layer initializes four smooth sigmoids whose slopes gradually sharpen during training. As they become steeper, these sigmoids effectively divide the input charge into distinct bins.

The mean of each sigmoid is a trainable parameter. During training, the model simultaneously learns:
	1.	The optimal sigmoid means (i.e., the charge thresholds / bin boundaries), and
	2.	The model weights that minimize the loss.

## What is the procedure that I have to follow in order to recreate this optimization process?
The workflow consists of two main training stages:

**Part 1 — Learn the optimal charge thresholds**

Train the model with the Soft Quantize Layer on a dataset injected with Gaussian noise
(mean = 0 e, sigma = 80 e).
This dataset must not be standardized, log-compressed, digitized, or quantized in any way.

**Part 2 — Train on 2-bit digitized inputs**

Train the same model architecture, but without the Soft Quantize Layer, using the same dataset digitized to 2 bits using the three thresholds learned in Part 1.
This dataset should again have no standardization or log compression, but its values should be digitized to 0.0, 1.0, 2.0, and 3.0.

To run the optimization process, you just need to change some arguments in `two_bit_optimization.ipynb`, then run all cells. The arguments that require change are:
* `dataset_dir`: Where your dataset is located. It should have `train`, `test`, `train_contained`, `test_contained` like on CERNbox
* `weights_dir`: The notebook will save the part-1 training checkpoint directory and the part-2 training checkpoint directory here.
* `performance_dir`: The notebook will save the final parquet file here, which will contain the performance variables (residuals_x, sigmacotB, etc.) of the best model tested on the test set.
* `model_type`: The model that you are training with. For non-quantized models, we have Conv2D_Max, Conv2D_Full, Conv2D_Slim, Conv1D_Full, Conv1D_Slim, Mlp_Full, and Mlp_Slim. For quantized models, you just need to add a "Q" to the front (ex: QConv2D_Full or QMlp_Slim).
* `tfrecords_exist`: Set to `False` if you previously have not generated the TFRecords with this notebook. If you are re-running the training pipeline and using the same TFRecords or you are training with a similar model (Full and Max both use the same TFRecords of 4 labels while Slim only uses 3 labels for its TFRecords), then you can set to `True` to save some time. Note: if you are running 2 or more copies of the notebook and generating the TFRrecords from scratch in parallel, it will overwrite one another. Therefore, make sure you finish generating the TFRecords for the first notebook that you set the flag to `False`, then set to `True` for the next notebooks you run in parallel.

All other parameters are fixed intentionally to maintain consistency across model trainings. While they are exposed as notebook arguments for flexibility, they should generally be left unchanged unless you explicitly know why you need to modify them.

---
## How can I extract the charge thresholds from part 1? 
`two_bit_optimization.ipynb` automatically extracts and prints the values of the charge thresholds. However, most of us want to just click "run all cells" and exit the notebook to let it run in the background to do its thing. However, you can use `get_best_thresholds()` located in `two_bit_optimization_helpers/train.py` to print out these thresholds. Just start up a fresh notebook or script, feed it the path of the part-1 checkpoints directory, model type, threshold offset, etc. and it will return the thresholds and levels. An example is shown in `training_tracker.ipynb` on how to import the function to the notebook (first cell) and use it (last cell).

## What if I want to test on dataset_2s?
`two_bit_optimization.ipynb` is only restricted to training, validating, and testing on dataset_3sr so far. We need to test on dataset_2s because that follows a more "realistic" distribution of the physics variables (x ,y, cotAlpha, cotBeta). I still need to add the functionality to automatically generate the TFRecords and test on dataset_2s but as of now, you will have to run a separate notebook to generate these TFRecords, then test on them. `dataset_2s_TFR.ipynb` allows you to generate TFRecords for dataset_2s for full precision TFRecords (make sure you set `to_standardize=True` and `log_compression=True`) and for TFRecords used to be loaded into 2-bits later (make sure you set `to_standardize=False` and `log_compression=False`). Also, for dataset_2s, we are scaling the labels/truth information of the TFRecords with the same scaling factor as the ones used on the validation/test set of dataset_3src. Just make sure for the slim models, you generate the TFRecords with the manual scaling of only (x, y, cotB) and for the full/max models, you generate the TFRecords with the manual scaling of all 4 labels (x, y, cotA, cotB).

For testing and saving the parquet files, refer to `vars_from_weights.ipynb`. All the necessary imports and custom helper functions are used to save a performance parquet that is tested on dataset_2s (digitized).





