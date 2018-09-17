import numpy as np
from src.preprocessing import *

################################################################################
# data generator fed to model.fit_generator
# affine inputs and masks together ans intensity transformation on inputs only
################################################################################

# randomly select batch-inouts for image augmentation
def form_double_batch(X, y, batch_size):
    idx = np.random.randint(0, X.shape[0], int(batch_size))
    return X[idx], y[idx]

# generator yeild
def double_batch_generator(X, y, batch_size, affine_seq, intensity_seq):
    while True:
        x_batch, y_batch = form_double_batch(X, y, batch_size)
        affine_seq_det = affine_seq.to_deterministic()
        new_x_batch = affine_seq_det.augment_images(x_batch*255)
        new_x_batch = intensity_seq.augment_images(new_x_batch)
        new_y_batch = affine_seq_det.augment_images(y_batch*255)
        yield new_x_batch/255, new_y_batch/255
