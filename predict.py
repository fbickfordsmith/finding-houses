# SETUP ------------------------------------------------------------------------

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

# Set base filepath. Import helper functions.
path = '/content/drive/My Drive/finding-houses/'
sys.path.insert(0, path)
from helper_functions import * 

# DATA -------------------------------------------------------------------------

# Load input image.
x = np.array(load_img(path+'images/rgb.png', color_mode='rgb'))

# Pad input image so that width and height are multiples of 256px.
x_pad = np.pad(x, pad_width=(
    (0, 256-(x.shape[0]%256)), # axis 0
    (0, 256-(x.shape[1]%256)), # axis 1
    (0, 0))) # axis 2

# Split input image into 256x256px frames.
x_split = []
Ni = x_pad.shape[0] // 256
Nj = x_pad.shape[1] // 256
for i in range(Ni):
    for j in range(Nj):
        x_split.append(x_pad[(i*256):(i*256)+256, (j*256):(j*256)+256])
x_split = np.array(x_split)

# PREDICTIONS ------------------------------------------------------------------

# Load trained model.
model = tf.keras.models.load_model(path+'training/model.h5')

# Compute predictions. Merge into single image. Crop back to original size.
yhat_split = model.predict(x_split.astype(np.float32))
yhat = np.concatenate(
    [np.concatenate(row, axis=1) for row in np.split(yhat_split, Ni, axis=0)],
    axis=0)

# Save predictions, cropped to input image size.
yhat = yhat[:x.shape[0], :x.shape[1]]
np.save(path+'predictions/train_valid_logit.npy', yhat)
np.save(path+'predictions/train_valid_proba.npy', sigmoid(yhat))
