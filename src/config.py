# encoding=utf8
import platform
import src.utils

################################################################################
# all the parameters used for project
################################################################################

# file path
if platform.system() == 'Windows':
    TRAIN_IMG_PATH = "E:\SaltSegmentation\\raw_data\\train\\images\\"
    TEST_IMG_PATH = "E:\SaltSegmentation\\raw_data\\test\\images\\"
    DEPTH_PATH = "E:\SaltSegmentation\\raw_data\\depths.csv"
    TRAIN_MASK_PATH = "E:\SaltSegmentation\\raw_data\\train\\masks\\"
    TRAIN_INFO_PATH = "E:\SaltSegmentation\\raw_data\\train.csv"
else:
    pass

#

# basic parameters
IMG_ORI_SIZE = 101
IMG_TAR_SIZE = 101

version = 1
BASIC_NAME = f'Unet_resnet_v{version}'
SAVE_MODEL_NAME = BASIC_NAME + '.model'
SUBMISSION_NAME = BASIC_NAME + '.csv'


# Keras Model parameters
START_NEURONS = 16
DROPOUT_RATIO = 0.5

MODEL1_ADAM_LR = 0.01
MODEL1_EPOCHS = 100
MODEL1_BATCH_SIZE = 64
MODEL1_STEPS_PER_EPOCH_TRAIN = 200
MODEL1_STEPS_PER_EPOCH_VAL = 100
MODEL1_LOSS = "binary_crossentropy"

MODEL2_ADAM_LR = 0.01
MODEL2_EPOCHS = 50
MODEL2_BATCH_SIZE = 64
MODEL2_STEPS_PER_EPOCH_TRAIN = 200
MODEL2_STEPS_PER_EPOCH_VAL = 100
MODEL2_LOSS = "lovasz_loss"

# ReduceLROnPlateau parameters
MODEL1_REDUCE_FACTOR = 0.5
MODEL1_REDUCE_PATIENT = 5

MODEL2_REDUCE_FACTOR = 0.5
MODEL2_REDUCE_PATIENT = 5
