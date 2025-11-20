
import os
from OptimizedDataGenerator_v2p5 import *

def generate_tfrecords(
    dataset_dir, 
    train_batch_size, 
    val_batch_size,
    to_standardize=False, 
    log_compression=False, 
    select_contained=False, 
    two_bit_optimized=False, 
    noise=-1, 
    min_threshold=None, 
    max_threshold=None, 
    timeslices=2, 
    labels_list=['x-midplane','y-midplane','cotAlpha','cotBeta'],
    seed=10, 
    max_workers=1, 
    tfrecords_exist=False, 
    model_type=None
):
    
    # determine the time stamps to use
    if timeslices==2:
        time_stamps = [0,19]
    elif timeslices==20:
        time_stamps = -1
    
    # determine selections
    contained_label=''
    if select_contained:
        contained_label='_contained'
        select_contained=True
    two_bit_optimized_label=''
    if two_bit_optimized:
        two_bit_optimized_label=f'_{model_type}_2bit_optimized'
    
    # determine input directories of parquets and output directories of tfrecords
    dataset_train_dir = os.path.join(dataset_dir, f"train{contained_label}{two_bit_optimized_label}")
    dataset_validation_dir = os.path.join(dataset_dir, f"test{contained_label}{two_bit_optimized_label}")

    tfrecords_dir = os.path.join(dataset_dir, "TFR_files", f"{timeslices}t")
    if noise == -1:
        tfrecords_dir_train = os.path.join(tfrecords_dir, f"TFR_train{contained_label}{two_bit_optimized_label}")
        tfrecords_dir_val   = os.path.join(tfrecords_dir, f"TFR_val{contained_label}{two_bit_optimized_label}")
    elif noise == [0,80]:
        tfrecords_dir_train = os.path.join(tfrecords_dir, f"TFR_train{contained_label}_noise-80e{two_bit_optimized_label}")
        tfrecords_dir_val   = os.path.join(tfrecords_dir, f"TFR_val{contained_label}_noise-80e{two_bit_optimized_label}")
    else:
        print('Invalid argument for noise. Either noise=-1 or noise=[0,80]')
        return 0

    dirs_to_create = [
    tfrecords_dir_train,
    tfrecords_dir_val,
    ]
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

    if tfrecords_exist:
        training_generator = OptimizedDataGenerator(
            dataset_base_dir = dataset_train_dir,
            file_type = "parquet",
            data_format = "3D",
            batch_size = train_batch_size,
            file_count = len(os.listdir(dataset_train_dir)),
            to_standardize = to_standardize, 
            log_compression = log_compression, 
            select_contained = select_contained,
            noise = noise,
            min_threshold = min_threshold,
            max_threshold = max_threshold,
            include_y_local= False,
            labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            input_shape = (timeslices,16,16),
            transpose = (0,2,3,1),
            shuffle = False,
            seed=seed,  

            tfrecords_dir = tfrecords_dir_train,
            use_time_stamps = time_stamps,
            max_workers = max_workers,
            load_from_tfrecords_dir = tfrecords_dir_train
        )

        validation_generator = OptimizedDataGenerator(
            dataset_base_dir = dataset_validation_dir,
            file_type = "parquet",
            data_format = "3D",
            batch_size = val_batch_size,
            file_count = len(os.listdir(dataset_validation_dir)),
            to_standardize = to_standardize,
            log_compression = log_compression,
            select_contained = select_contained,
            noise = noise,
            min_threshold = min_threshold,
            max_threshold = max_threshold,
            include_y_local= False,
            labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            input_shape = (timeslices,16,16),
            transpose = (0,2,3,1),
            shuffle = False, 
            files_from_end = True,
            seed=seed,

            tfrecords_dir = tfrecords_dir_val,
            use_time_stamps = time_stamps,
            max_workers = max_workers,
            load_from_tfrecords_dir = tfrecords_dir_val
        )
    
    else:
        training_generator = OptimizedDataGenerator(
            dataset_base_dir = dataset_train_dir,
            file_type = "parquet",
            data_format = "3D",
            batch_size = train_batch_size,
            file_count = len(os.listdir(dataset_train_dir)),
            to_standardize = to_standardize, 
            log_compression = log_compression, 
            select_contained = select_contained,
            noise = noise,
            min_threshold = min_threshold,
            max_threshold = max_threshold,
            include_y_local= False,
            labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            input_shape = (timeslices,16,16),
            transpose = (0,2,3,1),
            shuffle = False,
            seed=seed,  

            tfrecords_dir = tfrecords_dir_train,
            use_time_stamps = time_stamps,
            max_workers = max_workers,
        )
        
        validation_generator = OptimizedDataGenerator(
            dataset_base_dir = dataset_validation_dir,
            file_type = "parquet",
            data_format = "3D",
            batch_size = val_batch_size,
            file_count = len(os.listdir(dataset_validation_dir)),
            to_standardize = to_standardize,
            log_compression = log_compression,
            select_contained = select_contained,
            noise = noise,
            min_threshold = min_threshold,
            max_threshold = max_threshold,
            include_y_local= False,
            labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            input_shape = (timeslices,16,16),
            transpose = (0,2,3,1),
            shuffle = False, 
            files_from_end = True,
            seed=seed,

            tfrecords_dir = tfrecords_dir_val,
            use_time_stamps = time_stamps,
            max_workers = max_workers,
        ) 

    return dataset_train_dir, dataset_validation_dir, tfrecords_dir_train, tfrecords_dir_val

def load_tfrecords(tfrecords_dir_train, tfrecords_dir_val, seed=10, quantize=False, shuffle=True):
    training_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_dir_train,
    shuffle = shuffle,
    seed = seed,
    quantize = quantize
    )

    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir = tfrecords_dir_val,
        shuffle = shuffle,
        seed = seed,
        quantize = quantize
    )

    return training_generator, validation_generator