# encoding=utf8
import numpy as np
import pandas as pd
from functools import partial
import cv2
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
from src.preprocessing import *
from src.utils.pytorch.utils import * 
from src.config import *

################################################################################
# data generator fed to model.fit_generator
# affine inputs and masks together ans intensity transformation on inputs only
################################################################################

def form_double_batch(X, y, batch_size):
    '''
    randomly select batch-inouts for image augmentation
    :param X:
    :param y:
    :param batch_size:
    :return:
    '''
    idx = np.random.randint(0, X.shape[0], int(batch_size))

    return X[idx], y[idx]

def double_batch_generator(X, y, batch_size, affine_seq, intensity_seq):
    '''
    generator yeild
    :param X: input images
    :param y: label images
    :param batch_size: processing batch size
    :param affine_seq: pre-defined affine transformation augmentation methods
    :param intensity_seq: pre-defined intensity transformation augmentation methods
    :return: data generator
    '''
    while True:
        x_batch, y_batch = form_double_batch(X, y, batch_size)
        affine_seq_det = affine_seq.to_deterministic()
        new_x_batch = affine_seq_det.augment_images(x_batch*255)
        new_x_batch = intensity_seq.augment_images(new_x_batch)
        new_y_batch = affine_seq_det.augment_images(y_batch*255)
        yield new_x_batch/255, new_y_batch/255


class ShipDataset(Dataset):
    def __init__(self, data, transform=None, mode='train'):
        if mode == 'train' or mode == 'valid':
            self.x = data[0]
            self.y = data[1]
        elif mode == 'test':
            self.data = data
        else:
            raise RuntimeError('MODE_ERROR')
        self.transform = transform
        self.mode = mode
        self.pad_method = PAD_METHOD
        self.pad_size = (IMG_TAR_SIZE - IMG_ORI_SIZE) // 2

        if FIT_METHOD == 'resize_pad':
            self.aug_func_eval = partial(resize_pad_seq_eval, self.pad_size)
        elif FIT_METHOD == 'resize':
            self.aug_func_eval = resize_seq_eval

        if AUG:
            if FIT_METHOD == 'resize_pad':
                self.aug_func = partial(resize_pad_seq, self.pad_size)
            elif FIT_METHOD == 'resize':
                self.aug_func = resize_seq

        if INPUT_CHANNEL == 3:
            self.depth = np.tile(np.linspace(0, 1, IMG_TAR_SIZE), [IMG_TAR_SIZE, 1]).T

    def __len__(self):
        if self.mode == 'train' or self.mode == 'valid':
            return len(self.x)
        elif self.mode == 'test':
            return len(self.data)
        else:
            raise RuntimeError('MODE_ERROR')

    def __getitem__(self, idx):
        if self.mode == 'train':
            if AUG:
                resize_seq_det = self.aug_func().to_deterministic()
                new_x_batch = resize_seq_det.augment_image(self.x[idx])
                new_x_batch = intensity_seq.augment_image(new_x_batch) / 255
                new_y_batch = resize_seq_det.augment_image(self.y[idx]) / 255
            else:
                resize_seq_det = self.aug_func_eval().to_deterministic()
                new_x_batch = resize_seq_det.augment_image(self.x[idx]) / 255
                new_y_batch = resize_seq_det.augment_image(self.y[idx]) / 255
            if INPUT_CHANNEL == 3:
                new_x_batch = np.tile(new_x_batch, (1, 1, 3))
                new_x_batch = add_depth_channels(new_x_batch, self.depth)
            return new_x_batch, new_y_batch
        elif self.mode == 'valid':
            resize_seq_det = self.aug_func_eval().to_deterministic()
            new_x_batch = resize_seq_det.augment_image(self.x[idx]) / 255
            new_y_batch = resize_seq_det.augment_image(self.y[idx]) / 255
            if INPUT_CHANNEL == 3:
                new_x_batch = np.tile(new_x_batch, (1, 1, 3))
                new_x_batch = add_depth_channels(new_x_batch, self.depth)
            return new_x_batch, new_y_batch
        elif self.mode == 'test':
            resize_seq_det = self.aug_func_eval()
            test_data = resize_seq_det.augment_image(self.data[idx]) / 255
            if INPUT_CHANNEL == 3:
                test_data = np.tile(test_data, (1, 1, 3))
                new_x_batch = add_depth_channels(test_data, self.depth)
            return test_data
        else:
            raise RuntimeError('MODE_ERROR')


def make_loader(data, batch_size, num_workers=4, shuffle=False, transform=None, mode='train'):
    return DataLoader(
        dataset=ShipDataset(data, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
