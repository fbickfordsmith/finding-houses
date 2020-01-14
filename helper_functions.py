import numpy as np

def random_crop_flip(x_in, y_in, i0=None, j0=None, crop_shape=(256, 256)):
    # Sample frame from random location in image. Randomly flip frame.
    if i0 == None:
        i0 = np.random.randint(low=0, high=(x_in.shape[0]-crop_shape[0]))
    if j0 == None:
        j0 = np.random.randint(low=0, high=(x_in.shape[1]-crop_shape[1]))
    x_out = x_in[i0:(i0+crop_shape[0]), j0:(j0+crop_shape[1])]
    y_out = y_in[i0:(i0+crop_shape[0]), j0:(j0+crop_shape[1])]
    if np.random.uniform() < 0.5:
        x_out = np.flip(x_out, axis=0)
        y_out = np.flip(y_out, axis=0)
    if np.random.uniform() < 0.5:
        x_out = np.flip(x_out, axis=1)
        y_out = np.flip(y_out, axis=1)
    return x_out, y_out

def sample_training_data(x_train, y_train, num_examples=1000):
    # Sample set of frames from x_train and y_train.
    x, y = [], []
    for i in range(num_examples):
        xi, yi = random_crop_flip(x_train, y_train)
        x.append(xi)
        y.append(yi)
    return np.array(x), np.array(y)

def sigmoid(x):
    # Compute numerically stable sigmoid.
    return np.exp(-np.logaddexp(0, -x))
