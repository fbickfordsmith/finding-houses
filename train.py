# SETUP ------------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img

# Set base filepath. Import helper functions.
path = '/content/drive/My Drive/finding-houses/'
sys.path.insert(0, path)
from helper_functions import *

# DATA -------------------------------------------------------------------------

# Load training data. Convert label image to one-hot encoding.
x_all = np.array(load_img(path+'images/rgb.png', color_mode='rgb'))
y_all = np.array(load_img(path+'images/gt.png', color_mode='grayscale')) // 255
y_all = tf.keras.utils.to_categorical(y_all, num_classes=2, dtype=np.uint8)

# Reserve 256px-high strip of training images for validation.
x_train = x_all[256:]
x_valid = x_all[:256]
y_train = y_all[256:]
y_valid = y_all[:256]

# Crop x_valid and y_valid at the red line shown in `x_valid_uncropped.png`.
j0 = (x_valid.shape[1] % 256) + 256
x_valid = x_valid[:, j0:]
y_valid = y_valid[:, j0:]

# Split x_valid and y_valid into 256x256px frames.
num_splits = x_valid.shape[1] // 256
x_valid = np.array(np.split(x_valid, num_splits, axis=1))
y_valid = np.array(np.split(y_valid, num_splits, axis=1))

# Sample training frames. Bundle validation data.
x, y = sample_training_data(x_train, y_train, num_examples=40000)
xy_valid = (x_valid, y_valid)

# MODEL ------------------------------------------------------------------------

# Implement model as specified in instructions.
model = tf.keras.models.Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(256, 256, 3)),
    Activation('relu'),
    Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    UpSampling2D(size=(2, 2)),
    Conv2D(filters=2, kernel_size=(5, 5), padding='same')])

# Use binary cross-entropy loss and Adam optimiser.
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

# TRAINING ---------------------------------------------------------------------

# Train model for 50 epochs.
history = model.fit(x, y, batch_size=256, epochs=50, validation_data=xy_valid)

# Save trained model and in-training metric values.
model.save(path+'training/model.h5')
metrics = pd.DataFrame({
    'loss_train':history.history['loss'],
    'loss_valid':history.history['val_loss'],
    'acc_train':history.history['accuracy'],
    'acc_valid':history.history['val_accuracy']})
metrics.to_csv(path+'training/metrics.csv', index=False)
