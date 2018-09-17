# encoding=utf8
import numpy as np
import pandas as pd
import cv2
from src.utils import upsample
from src.config import *
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

################################################################################
# load images and depth info from input file
# train_df: id, z, images, masks
# test_df: id, z, images
################################################################################


## convert salt coverage to class
def _cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


## used to load data from data files
class DataLoader:

    def __init__(self):
        train_df, self.test_df = self._load_depth()
        self._load_image_mask(train_df, self.test_df)
        train_df["coverage"] = train_df.masks.map(np.sum) / pow(IMG_ORI_SIZE, 2)
        train_df["coverage_class"] = train_df.coverage.map(_cov_to_class)
        self.x_train, self.x_valid, self.y_train, self.y_valid = self._get_train_test_split(train_df)
        train_df = None
        self.x_test = np.array(self.test_df.images.tolist()).reshape(-1, IMG_TAR_SIZE, IMG_TAR_SIZE, 1)

    def _load_image_mask(self, train_df, test_df):
        # load image data & mask data
        train_df['images'] = [np.array(cv2.imread(TRAIN_IMG_PATH + "{}.png".format(idx), 0)) for idx in tqdm_notebook(train_df.index)]
        train_df['masks'] = [np.array(cv2.imread(TRAIN_MASK_PATH + "{}.png".format(idx), 0)) for idx in tqdm_notebook(train_df.index)]
        test_df['images'] = [np.array(cv2.imread(TEST_IMG_PATH + "{}.png".format(idx), 0)) for idx in tqdm_notebook(test_df.index)]
        # Normalize image vectors
        train_df['images'] /= 255
        test_df['images'] /= 255
        train_df['masks'] /= 255

    @staticmethod
    def _load_depth():
        train_df = pd.read_csv(TRAIN_INFO_PATH, index_col="id", usecols=[0])
        depths_df = pd.read_csv(DEPTH_PATH, index_col="id")
        depths_df['z'] = depths_df['z'].astype('float')
        train_df = train_df.join(depths_df)
        test_df = depths_df[~depths_df.index.isin(train_df.index)]
        return train_df, test_df

    ## get train & validation split stratified by salt coverage
    def _get_train_test_split(self, train_df):
        x_train, x_valid, y_train, y_valid = train_test_split(
            np.array(train_df.images.map(upsample).tolist()).reshape(-1, IMG_TAR_SIZE, IMG_TAR_SIZE, 1),
            np.array(train_df.masks.map(upsample).tolist()).reshape(-1, IMG_TAR_SIZE, IMG_TAR_SIZE, 1),
            test_size=0.2, stratify=train_df.coverage_class, random_state=1234)
        return x_train, x_valid, y_train, y_valid

    def get_train_x(self):
        return self.x_train, self.x_valid

    def get_train_y(self):
        return self.y_train, self.y_valid

    def get_test_x(self):
        return self.x_test

    def get_test_df(self):
        return self.test_df
