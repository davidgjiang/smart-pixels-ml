#!/usr/bin/env python3
import numpy as np
import pandas as pd
import glob
import os
import shutil
from tqdm import tqdm
from natsort import natsorted

def preselection_processing(input_directory, output_directory, file_type='parquet'):
    # Collect all simulation files
    simulation_files = natsorted(glob.glob(os.path.join(input_directory, f"*.{file_type}")))
    filenames = [os.path.basename(f) for f in simulation_files]

    # Clear or create the output directory
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)

    # Process each file
    for i in tqdm(range(len(simulation_files)), desc="Processing files..."):
        temp_df = pd.read_parquet(simulation_files[i])
        filtered_df = temp_df[temp_df['chargeOriginal_atEdge'] < 50]
        final_df = filtered_df.reset_index(drop=True)

        # Save to new directory
        output_path = os.path.join(output_directory, filenames[i])
        final_df.to_parquet(output_path)

if __name__ == '__main__':
    print('*** Preselection Processor ***')
    preselection_processing(input_directory='/data/dajiang/smart-pixels/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/test/', 
                            output_directory='/data/dajiang/smart-pixels/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/test_contained/')
    preselection_processing(input_directory='/data/dajiang/smart-pixels/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/train/', 
                            output_directory='/data/dajiang/smart-pixels/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/train_contained/')
    print('Processing done for 3sr.')
    preselection_processing(input_directory='/data/dajiang/smart-pixels/largerWindowPreliminary/dataset_2s_16x16_50x12P5_centeredIncidence_parquets/test/', 
                            output_directory='/data/dajiang/smart-pixels/largerWindowPreliminary/dataset_2s_16x16_50x12P5_centeredIncidence_parquets/test_contained/')
    print('Processing done for 2s.')
