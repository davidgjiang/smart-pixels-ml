import tensorflow as tf
import numpy as np
import random
import os
import shutil
import gc
from AnnealingScheduler import *
from loss import (
    custom_loss,
    custom_diag_loss,
    custom_sse_loss
)
from models.conv2d_model_nonquantized import (
    Conv2D_Max,
    Conv2D_Full,
    Conv2D_Slim,
    Conv2D_Max_SoftQuantizer,
    Conv2D_Full_SoftQuantizer,
    Conv2D_Slim_SoftQuantizer,
)
from models.conv1d_model_nonquantized import (
    Conv1D_Full,
    Conv1D_Slim,
    Conv1D_Full_SoftQuantizer,
    Conv1D_Slim_SoftQuantizer,
)
from models.mlp_encoder_model_nonquantized import (
    Mlp_Full,
    Mlp_Slim,
    Mlp_Full_SoftQuantizer,
    Mlp_Slim_SoftQuantizer,
)
from models.conv2d_model_quantized import (
    QConv2D_Max,
    QConv2D_Full,
    QConv2D_Slim,
    QConv2D_Max_SoftQuantizer,
    QConv2D_Full_SoftQuantizer,
    QConv2D_Slim_SoftQuantizer,
)
from models.conv1d_model_quantized import (
    QConv1D_Full,
    QConv1D_Slim,
    QConv1D_Full_SoftQuantizer,
    QConv1D_Slim_SoftQuantizer,
)
from models.mlp_encoder_model_quantized import (
    QMlp_Full,
    QMlp_Slim,
    QMlp_Full_SoftQuantizer,
    QMlp_Slim_SoftQuantizer,
)
model_list = {
    'Conv2D_Max': [Conv2D_Max, Conv2D_Max_SoftQuantizer],
    'Conv2D_Full': [Conv2D_Full, Conv2D_Full_SoftQuantizer],
    'Conv2D_Slim': [Conv2D_Slim, Conv2D_Slim_SoftQuantizer],

    'Conv1D_Full': [Conv1D_Full, Conv1D_Full_SoftQuantizer],
    'Conv1D_Slim': [Conv1D_Slim, Conv1D_Slim_SoftQuantizer],

    'Mlp_Full': [Mlp_Full, Mlp_Full_SoftQuantizer],
    'Mlp_Slim': [Mlp_Slim, Mlp_Slim_SoftQuantizer],

    'QConv2D_Max': [QConv2D_Max, QConv2D_Max_SoftQuantizer],
    'QConv2D_Full': [QConv2D_Full, QConv2D_Full_SoftQuantizer],
    'QConv2D_Slim': [QConv2D_Slim, QConv2D_Slim_SoftQuantizer],

    'QConv1D_Full': [QConv1D_Full, QConv1D_Full_SoftQuantizer],
    'QConv1D_Slim': [QConv1D_Slim, QConv1D_Slim_SoftQuantizer],

    'QMlp_Full': [QMlp_Full, QMlp_Full_SoftQuantizer],
    'QMlp_Slim': [QMlp_Slim, QMlp_Slim_SoftQuantizer],
}

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

    # Construct model
    if soft_quantize_layer:
        if 'Conv2D' in model_type:
            return model_list[model_type][1](
                (16, 16, timeslices),
                n_filters=5,
                pool_size=3,
                initial_thresholds=initial_thresholds,
                threshold_offset=threshold_offset,
                initial_levels=initial_levels,
            )
        else:
            return model_list[model_type][1](
                (16,16,timeslices),
                initial_thresholds=initial_thresholds,
                threshold_offset=threshold_offset,
                initial_levels=initial_levels,
            )
    else:
        if 'Conv2D' in model_type:
            return model_list[model_type][0](
                (16, 16, timeslices),
                n_filters=5,
                pool_size=3,
            )
        else: 
            return model_list[model_type][0]((16, 16, timeslices))

    

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
        print('Please specify train_type: soft_quantize_layer or 2bit_optimized.')
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
    
def get_thresholds(
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
    if 'Conv2D' in model_type:
        model = model_list[model_type][1](
            (16, 16, timeslices),
            n_filters=5,
            pool_size=3,
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    else:
        model = model_list[model_type][1](
            (16,16,timeslices),
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    
    # Find the best checkpoint (lowest validation loss)
    checkpoints_contents = os.listdir(checkpoints)
    vlosses = [float(c.split('-v')[1].split('.hdf5')[0]) for c in checkpoints_contents]
    bestfile = checkpoints_contents[np.argmin(vlosses)]
    model.load_weights(checkpoints+'/'+bestfile)
    print('Best model: {}'.format(bestfile))

    # Extract the thresholds of the best checkpoint
    layer = model.get_layer('soft_quantizer_output')
    print('---Soft Quantize Layer Details---')
    thresholds = layer.thresholds.numpy()
    levels = layer.levels.numpy()
    print(f'  Thresholds (e): {thresholds}') 
    print(f'  Levels: {levels}')

    return thresholds, levels

def cleanup_models_and_generators(objects: list):
    '''
    Deletes models and generators, clears TF session, and forces garbage collection.
    
    Args:
        objects (list): List of models, generators, or other large objects to delete.
    '''
    for obj in objects:
        try:
            # If the object has large internal data, clear it first
            if hasattr(obj, 'current_dataframes'):
                obj.current_dataframes = None
            if hasattr(obj, 'files'):
                obj.files = None

            del obj
        except Exception as e:
            print(f'Warning: could not delete object {obj}: {e}')

    # Clear TF/Keras session to free GPU memory
    tf.keras.backend.clear_session()
    
    # Force Python garbage collection
    gc.collect()
    
    print('Cleanup complete: models, generators, and GPU memory freed.')