# encoding=utf8
import platform

################################################################################
# all the parameters used for project
################################################################################

# read data
with open('../config/config.yaml','r') as f:
    data = yaml.load(f)

DATA = data
# file path
if platform.system() == 'Windows':
    TRAIN_IMG_PATH = data['windows_path']['TRAIN_IMG_PATH']
    TEST_IMG_PATH = data['windows_path']['TEST_IMG_PATH ']
    DEPTH_PATH = data['windows_path']['DEPTH_PATH']
    TRAIN_MASK_PATH = data['windows_path']['TRAIN_MASK_PATH']
    TRAIN_INFO_PATH = data['windows_path']['TRAIN_INFO_PATH']
    RESUME_PATH = data['windows_path']['RESUME_PATH']
else:
    TRAIN_IMG_PATH = data['linux_path']['TRAIN_IMG_PATH']
    TEST_IMG_PATH = data['linux_path']['TEST_IMG_PATH ']
    DEPTH_PATH = data['linux_path']['DEPTH_PATH']
    TRAIN_MASK_PATH = data['linux_path']['TRAIN_MASK_PATH']
    TRAIN_INFO_PATH = data['linux_path']['TRAIN_INFO_PATH']
    RESUME_PATH = data['linux_path']['RESUME_PATH']

# basic parameters
IMG_ORI_SIZE = data['basic_para']['IMG_ORI_SIZE']
IMG_TAR_SIZE = data['basic_para']['IMG_TAR_SIZE']

version = data['version']

# Model parameters
START_NEURONS = data['model_build']['START_NEURONS']
DROPOUT_RATIO = data['model_build']['DROPOUT_RATIO']

MODEL1_ADAM_LR = data['model_train'][0]['MODEL_ADAM_LR']
MODEL1_EPOCHS = data['model_train'][0]['MODEL_EPOCHS']
MODEL1_BATCH_SIZE = data['model_train'][0]['MODEL_BATCH_SIZE']
MODEL1_STEPS_PER_EPOCH_TRAIN = data['model_train'][0]['MODEL_STEPS_PER_EPOCH_TRAIN']
MODEL1_STEPS_PER_EPOCH_VAL = data['model_train'][0]['MODEL_STEPS_PER_EPOCH_VAL']
MODEL1_LOSS = data['model_train'][0]['MODEL_LOSS']

MODEL2_ADAM_LR = data['model_train'][1]['MODEL_ADAM_LR']
MODEL2_EPOCHS = data['model_train'][1]['MODEL_EPOCHS']
MODEL2_BATCH_SIZE = data['model_train'][1]['MODEL_BATCH_SIZE']
MODEL2_STEPS_PER_EPOCH_TRAIN = data['model_train'][1]['MODEL_STEPS_PER_EPOCH_TRAIN']
MODEL2_STEPS_PER_EPOCH_VAL = data['model_train'][1]['MODEL_STEPS_PER_EPOCH_VAL']
MODEL2_LOSS = data['model_train'][1]['MODEL_LOSS']

# ReduceLROnPlateau parameters
MODEL1_REDUCE_FACTOR = data['R_LR'][0]['MODEL_REDUCE_FACTOR']
MODEL1_REDUCE_PATIENT = data['R_LR'][0]['MODEL_REDUCE_PATIENT']

MODEL2_REDUCE_FACTOR = data['R_LR'][1]['MODEL_REDUCE_FACTOR']
MODEL2_REDUCE_PATIENT = data['R_LR'][1]['MODEL_REDUCE_PATIENT']

# mode
TRAIN = 1
TEST = 2
MODE = TRAIN

# aug
AUG = data['augmentation']['AUG']
FIT_METHOD = data['augmentation']['FIT_METHOD']
PAD_METHOD = data['augmentation']['PAD_METHOD']
KFOLD = data['augmentation']['KFOLD']

# thresholds
THRESHOLD_BEST = data['THRESHOLD_BEST']

# submission
SUBMISSION = data['SUBMISSION']

# library
LIBRARY = data['LIBRARY']

# Name
BASIC_NAME = f'Unet_resnext50_v{version}'
SAVE_MODEL_NAME = BASIC_NAME + '.model'
SUBMISSION_NAME = BASIC_NAME + '.csv'

# load from checkpoint
LOAD_CHECKPOINT = data['CHECKPOINT']


def write_new_yaml():
    DATA['version'] = version+1
    DATA['THRESHOLD_BEST'] = THRESHOLD_BEST
    basic_yaml_name = f'../config/config_v{version}'
    save_yaml_name = basic_yaml_name + '.yaml'

    with open(save_yaml_name,'w') as f:
        yaml.dump(DATA, f, default_flow_style=False)
