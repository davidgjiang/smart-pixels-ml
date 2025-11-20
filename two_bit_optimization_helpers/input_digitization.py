import glob
import os
import pandas as pd
import numpy as np
from dataset_utils import *
from models.conv2d_model_nonquantized import (
    Conv2D_Max,
    Conv2D_Full,
    Conv2D_Max_SoftQuantizer,
    Conv2D_Full_SoftQuantizer,
)
from models.mlp_encoder_model_nonquantized import (
    Mlp_Full,
    Mlp_Slim,
    Mlp_Full_SoftQuantizer,
    Mlp_Slim_SoftQuantizer,
)

def save_quantized_manual_parquet(
    input_filepath, 
    output_dir, 
    charge_thresholds, 
    quant_values, 
    shuffled=True, 
    noise=-1, 
    min_threshold=None,
    max_threshold=None
):
    """
    charge_levels = list of 3 values indicating the charge boundaries for the 2-bit binnings
    quant_values = list of 4 values representing the output value that the bins will be mapped to
    noise = add gaussian noise in the form of [mu, sigma] (if -1 then no noise will be added)
    threshold = zero all charges <= threshold value (if -1 then no threshold will be added)
    shuffled = boolean representing if the dataset is shuffled or unshuffled
    """
    temp_df = pd.read_parquet(input_filepath)
    if noise != -1:
        temp_df = add_noise(x=temp_df, mu=noise[0], sig=noise[1], shuffled=shuffled, seed=None)
    if min_threshold != None:
        temp_df = apply_threshold(x=temp_df, thresh=min_threshold, minimum=True, shuffled=shuffled)
    if max_threshold != None:
        temp_df = apply_threshold(x=temp_df, thresh=max_threshold, minimum=False, shuffled=shuffled)
    temp_quantized_df = quantize_manual(x=temp_df, charge_levels=charge_thresholds, quant_values=quant_values, shuffled=shuffled)
    output_filepath = output_dir + input_filepath.split('/')[-1]
    temp_quantized_df.to_parquet(output_filepath)
    print(f'{output_filepath} successfully processed and saved.')

def process_inputs(
    checkpoints,
    dataset_train_dir,
    dataset_validation_dir, 
    model_type,
    timeslices=2,
    initial_thresholds=[400, 1000, 2000],
    threshold_offset=0.0,
    initial_levels=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    ):

    # Create model based on model type (it needs to be model w/ soft quantize layer)
    if model_type == "Conv2D_Max":
        model = Conv2D_Max_SoftQuantizer(
            (16, 16, timeslices),
            n_filters=5,
            pool_size=3,
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    elif model_type == "Conv2D_Full":
        model = Conv2D_Full_SoftQuantizer(
            (16, 16, timeslices),
            n_filters=5,
            pool_size=3,
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    elif model_type == "Mlp_Full":
        model = Mlp_Full_SoftQuantizer(
            (16, 16, timeslices),
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    elif model_type == "Mlp_Slim":
        model = Mlp_Slim_SoftQuantizer(
            (16, 16, timeslices),
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    
    # Find the best checkpoint (lowest validation loss)
    checkpoints_contents = os.listdir(checkpoints)
    vlosses = [float(c.split("-v")[1].split(".hdf5")[0]) for c in checkpoints_contents]
    bestfile = checkpoints_contents[np.argmin(vlosses)]
    model.load_weights(checkpoints+'/'+bestfile)
    print('Best model: {}'.format(bestfile))

    # Extract the thresholds of the best checkpoint
    layer = model.get_layer("soft_quantizer_output")
    print('---Soft Quantize Layer Details---')
    thresholds = layer.thresholds.numpy()
    levels = layer.levels.numpy()
    print(f"  Thresholds (e): {thresholds}") 
    print(f'  Levels: {levels}')

    # Process the parquet files
    train_files = glob.glob(dataset_train_dir+'/*.parquet')
    test_files = glob.glob(dataset_validation_dir+'/*.parquet')

    output_train_dir = f'{dataset_train_dir}_{model_type}_2bit_optimized'
    output_test_dir = f'{dataset_validation_dir}_{model_type}_2bit_optimized'
    dirs_to_create = [
        output_train_dir,
        output_test_dir
    ]
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

    for file in train_files:
        save_quantized_manual_parquet(input_filepath=file, output_dir=output_train_dir, charge_thresholds=thresholds, quant_values=levels)

    for file in test_files:
        save_quantized_manual_parquet(input_filepath=file, output_dir=output_test_dir, charge_thresholds=thresholds, quant_values=levels)

    return output_train_dir, output_test_dir