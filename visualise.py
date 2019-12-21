# SETUP ------------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

# Set base filepath. Import helper functions.
path = '/content/drive/My Drive/finding-houses/'
sys.path.insert(0, path)
from helper_functions import *

# DATA -------------------------------------------------------------------------

# Load input image, label image, predictions, in-training metric values.
x = np.array(load_img(path+'images/rgb.png', color_mode='rgb'))
y = np.array(load_img(path+'images/gt.png', color_mode='grayscale')) // 255
y = tf.keras.utils.to_categorical(y, num_classes=2, dtype=np.uint8)
yhat = np.load(path+'predictions/train_valid_proba.npy')
metrics = pd.read_csv(path+'training/metrics.csv')

# VISUALISATION ----------------------------------------------------------------

# Show in-training metric values.
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(metrics['loss_train'], label='training')
axes[0].plot(metrics['loss_valid'], label='validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(metrics['acc_train'], label='training')
axes[1].plot(metrics['acc_valid'], label='validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
plt.tight_layout()
plt.savefig(path+'training/metrics.png', dpi=300)

# Show validation data.
fig, axes = plt.subplots(4, 1, figsize=(10, 6))
axes[0].imshow(x[:256])
axes[0].set_title('Input')
axes[1].imshow(yhat[:256, :, 0])
axes[1].set_title('Prediction (probability)')
axes[2].imshow(yhat[:256, :, 0]>0.5)
axes[2].set_title('Prediction (binary)')
axes[3].imshow(y[:256, :, 0])
axes[3].set_title('Label')
[ax.axis('off') for ax in axes]
plt.tight_layout()
plt.savefig(path+'predictions/valid_comparison.png', dpi=300)

# Show predictions (probabilities) on whole input image.
plt.imshow(yhat[:, :, 0])
plt.axis('off')
plt.colorbar()
plt.savefig(path+'predictions/train_valid_proba.png', dpi=300)

# Show predictions (binary) on whole input image.
plt.imshow(yhat[:, :, 0]>0.5)
plt.axis('off')
plt.savefig(path+'predictions/train_valid_binary_rgb.png', dpi=300)

# Show predictions (binary, black and white) on whole input image.
plt.imshow(yhat[:, :, 0]>0.5, cmap='Greys')
plt.axis('off')
plt.savefig(path+'predictions/train_valid_binary_bw.png', dpi=300)
