import tensorflow as tf
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import keras
from .utils import *

BUFFER_SIZE = 500

def load_muscle_data(image_list, mask_list):
    ## Set Input Windows
    windows = ['soft','bone','custom']

    ## Create Tensorflow Datasets
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    # Apply the function to the dataset
    image_ds = dataset.map(tf_process_path_C2C).map(
        lambda img, mask: (tf.ensure_shape(img, (512,512)), tf.ensure_shape(mask, (512,512)))
    )
    processed_image_ds = image_ds.map(tf_preprocess_C2C).map(
        lambda img, mask: (tf.ensure_shape(img, (512,512,3)), tf.ensure_shape(mask, (512,512)))
    )

    # Split Datasets 80/10/10
    processed_img_ds_train, processed_image_ds_valtest = keras.utils.split_dataset(processed_image_ds, left_size=0.8)
    processed_image_ds_val, processed_image_ds_test = keras.utils.split_dataset(processed_image_ds_valtest, left_size=0.5)
    return processed_img_ds_train, processed_image_ds_val, processed_image_ds_test

def train_unet(model_input, train_val_data, epochs=1, batch_size=32, lr=1e-5, verbose=0, modelname=''):
  processed_img_ds_train, processed_image_ds_val = train_val_data

  logdir = os.path.join('logs', modelname)

  callbacks = [
  # EarlyStopping(monitor='val_loss', patience=2),
  ModelCheckpoint(filepath=f'model_{modelname}.keras', monitor='val_loss', save_best_only=True, verbose=1),
  TensorBoard(logdir, histogram_freq=1), 
  # ConfusionMatrixTensorBoard(
  #   log_dir=logdir,
  #   validation_data=val_dataset,  # Provide validation data here
  #   num_classes=num_classes_muscle)
  ]
  
  # batch the data
  train_dataset = processed_img_ds_train.cache().shuffle(BUFFER_SIZE).batch(batch_size)
  val_dataset = processed_image_ds_val.cache().shuffle(BUFFER_SIZE).batch(batch_size)

  # setting the learning rate 
  model_input.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), ## Define Loss Function
    loss=dice_coef_multilabel, ## Define Loss Function
    metrics=['accuracy']) # dice_coef_weighted_all_classes
  
  model_history = model_input.fit(train_dataset, epochs=epochs, verbose=verbose, callbacks=[callbacks],validation_data=val_dataset)

  show_predictions(model_input, train_dataset, 2)
  return model_input, model_history, train_dataset, val_dataset
  
