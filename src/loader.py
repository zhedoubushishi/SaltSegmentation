#encoding=utf8

import numpy as np
import pandas as pd
import * from config
from imgaug import augmenters as iaa



################################################################################
# load images and depth info from input file
# train_df: id, z, images, masks
# test_df: id, z, images
################################################################################


class DataLoader():

    def __init__():
        load_depth()
        load_image_mask()



    def load_image_mask():
        # load image data & mask data
        self.train_df['images'] = [np.array(load_img(TRAIN_IMG_PATH + "{}.png".format(idx),grayscale=True)) for idx in tqdm_notebook(train_df.index)]
        self.train_df['masks'] = [np.array(load_img(TRAIN_MASK_PATH + "{}.png".format(idx), grayscale=True)) for idx in tqdm_notebook(train_df.index)]
        self.test_df['images'] = [np.array(load_img(TEST_IMG_PATH + "{}.png".format(idx), grayscale=True)) for idx in tqdm_notebook(test_df.index)]
        # Normalize image vectors
        self.train_df['images'] = train_df['images'] / 255
        self.test_df['images'] = test_df['images'] / 255
        self.train_df['masks'] = train_df['masks'] / 255

    def load_depth():
        self.train_df = pd.read_csv(TRAIN_INFO_PATH., index_col="id", usecols=[0])
        depths_df = pd.read_csv(DEPTH_PATH, index_col="id")
        depths_df['z'] = self.depths_df['z'].astype('float')
        self.train_df = self.train_df.join(depths_df)
        self.test_df = depths_df[~depths_df.index.isin(self.train_df.index)]
