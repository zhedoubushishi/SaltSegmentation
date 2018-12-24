# encoding=utf8
import numpy as np
import pandas as pd
import cv2
from src.utils.keras.utils import upsample
from src.config import *
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

################################################################################
# load images and depth info from input file
# train_df: id, z, images, masks
# test_df: id, z, images
################################################################################


############################### keras version ########################################
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
        self.x_test = np.array(self.test_df.images.map(upsample).tolist()).reshape(-1, IMG_TAR_SIZE, IMG_TAR_SIZE, 1)

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


################################# pytorch version #######################################
## convert salt coverage to class
def cov_to_class_1(mask):
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0  # empty
    if cover == ((mask * outer) > 0.5).sum():
        return 1  # border
    if np.all(mask == mask[0]):
        return 2  # vertical

    percentage = cover / (101 * 101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


def cov_to_class_2(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


## used to load data from data files
class my_DataLoader:
    def __init__(self, train=False, test=False, Kfold=False, test_size=0.2):
        self.test = test
        self.train = train
        self.Kfold = Kfold
        self.test_size = test_size
        self.num_fold = int(1 / test_size)

        train_df, self.test_df = self._load_depth()

        if self.train:
            self._load_image_mask(train_df)
            train_df["coverage"] = train_df.masks.map(np.sum) / pow(IMG_ORI_SIZE, 2)
            train_df["coverage_class"] = train_df.masks.map(cov_to_class_1)
            self.x_train, self.x_valid, self.y_train, self.y_valid = self._get_train_test_split(train_df, self.Kfold,
                                                                                                self.num_fold)

        if self.test:
            test_df['images'] = self._load_image_test(self.test_df)
            self.x_test = np.array(self.test_df.images.tolist()).reshape(-1, IMG_ORI_SIZE, IMG_ORI_SIZE, 1)

    @staticmethod
    def _load_image_mask(train_df):
        # load image data & mask data
        train_df['images'] = [np.array(cv2.imread(TRAIN_IMG_PATH + "{}.png".format(idx), 0)) for idx in
                              tqdm_notebook(train_df.index)]
        train_df['masks'] = [np.array(cv2.imread(TRAIN_MASK_PATH + "{}.png".format(idx), 0)) for idx in
                             tqdm_notebook(train_df.index)]
        # Normalize image vectors
        # train_df['images'] /= 255
        # train_df['masks'] /= 255

    @staticmethod
    def _load_image_test(test_df):
        return [np.array(cv2.imread(TEST_IMG_PATH + "{}.png".format(idx), 0)) for idx in tqdm_notebook(test_df.index)]

    @staticmethod
    def _load_depth():
        train_df = pd.read_csv(TRAIN_INFO_PATH, index_col="id", usecols=[0])
        depths_df = pd.read_csv(DEPTH_PATH, index_col="id")
        depths_df['z'] = depths_df['z'].astype('float')
        train_df = train_df.join(depths_df)
        test_df = depths_df[~depths_df.index.isin(train_df.index)]
        return train_df, test_df

    ## get train & validation split stratified by salt coverage
    @staticmethod
    def _get_train_test_split(train_df, Kfold, num_fold):
        x_train, x_valid, y_train, y_valid = [], [], [], []
        skf = StratifiedKFold(n_splits=num_fold, random_state=1234, shuffle=True)
        for train_index, valid_index in skf.split(train_df.index.values, train_df.coverage_class):
            x_tr = np.array(train_df.images[train_index].tolist()).reshape(-1, IMG_ORI_SIZE, IMG_ORI_SIZE, 1)
            x_tr = np.append(x_tr, [np.fliplr(x) for x in x_tr], axis=0)
            x_train.append(x_tr)
            x_valid.append(np.array(train_df.images[valid_index].tolist()).reshape(-1, IMG_ORI_SIZE, IMG_ORI_SIZE, 1))
            y_tr = np.array(train_df.masks[train_index].tolist()).reshape(-1, IMG_ORI_SIZE, IMG_ORI_SIZE, 1)
            y_tr = np.append(y_tr, [np.fliplr(y) for y in y_tr], axis=0)
            y_train.append(y_tr)
            y_valid.append(np.array(train_df.masks[valid_index].tolist()).reshape(-1, IMG_ORI_SIZE, IMG_ORI_SIZE, 1))
            if not Kfold:
                break
        return x_train, x_valid, y_train, y_valid

    def get_train(self):
        return self.x_train, self.y_train

    def get_valid(self):
        return self.x_valid, self.y_valid

    def get_test_x(self):
        return self.x_test

    def get_test_df(self):
        return self.test_df
