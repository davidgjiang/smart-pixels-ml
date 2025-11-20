import numpy as np
import random
import os
import shutil
from AnnealingScheduler import *
from loss import (
    custom_loss,
    custom_diag_loss,
    custom_sse_loss
)
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


def create_model(
    model_type,
    timeslices=2,
    soft_quantize_layer=False,
    initial_thresholds=None,
    threshold_offset=0.0,
    initial_levels=None,
):
    if initial_thresholds is None:
        initial_thresholds = [400, 1000, 2000]
    if initial_levels is None:
        initial_levels = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    # Create model based on model type
    if not soft_quantize_layer:
        if model_type == "Conv2D_Max":
            model = Conv2D_Max(
                (16, 16, timeslices),
                n_filters=5,
                pool_size=3,
            )
        elif model_type == "Conv2D_Full":
            model = Conv2D_Full(
                (16, 16, timeslices),
                n_filters=5,
                pool_size=3,
            )
        elif model_type == "Mlp_Full":
            model = Mlp_Full((16, 16, timeslices))
        elif model_type == "Mlp_Slim":
            model = Mlp_Slim((16, 16, timeslices))

    else:  # soft_quantize_layer == True
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

    return model

import shutil

def train(
    model,
    model_type, 
    weights_directory,
    training_generator,
    validation_generator, 
    timeslices=2,
    train_type=None, # full_precision, soft_quantize_layer, 2bit_optimized
    epochs=5,
    seed=10, 
    verbose=1):

    random.seed(seed)

    if 'Max' in model_type:
        loss=custom_loss
    elif 'Full' in model_type:
        loss=custom_diag_loss
    elif 'Slim' in model_type:
        loss=custom_sse_loss

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
        loss=loss
    )
    model.summary()

    fingerprint = '%08x' % random.randrange(16**8)
    if train_type:
        weights_directory = weights_directory + '/weights-{}t-{}-{}-{}-checkpoints'.format(
            timeslices, model_type, train_type, fingerprint
        )
    else:
        print('Please specify train_type: full_precision, soft_quantize_layer, or 2bit_optimized.')
        return 0
    print(f'Checkpoints saved to {weights_directory}')
    
    # --- Clear old checkpoint directory if it exists ---
    if os.path.exists(weights_directory):
        shutil.rmtree(weights_directory)
    os.makedirs(weights_directory)  # Create fresh directory

    checkpoint_filepath = weights_directory + '/weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5'
    
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=False,
    )

    print('Model fingerprint: {}'.format(fingerprint))

    if train_type == 'soft_quantize_layer':
        scheduler_callback = AnnealingScheduler(
            schedule='cosine',  
            target_layer_name='soft_quantizer_output', 
            initial_k=1.0,
            final_k=67.0,
            verbose=1      
        )
    
        model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=[mcp, scheduler_callback],
            epochs=epochs,
            shuffle=False,
            verbose=verbose
        )
    else:
        model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=[mcp],
            epochs=epochs,
            shuffle=False,
            verbose=verbose
        )
    
    return weights_directory
    
