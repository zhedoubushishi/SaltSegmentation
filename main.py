# encoding=utf8
import numpy as np
import pandas as pd
from src.config import *
from src.loader import DataLoader
from src.utils import iou_metric_batch, downsample
from src.keras_models import KerasModel
from tqdm import tqdm_notebook


"""
used for converting the decoded image to rle mask
im: numpy array, 1 - mask, 0 - background
Returns run length as string formated
"""
def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_best_threshold(model, x_valid, y_valid):

    preds_valid = model.predict(x_valid, IMG_TAR_SIZE)
    # Scoring for last model, choose threshold by validation data
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori))

    ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    return threshold_best


def main():
    dl = DataLoader()
    x_train, x_valid = dl.get_train_x()
    y_train, y_valid = dl.get_train_y()
    x_test = dl.get_test_x()
    test_df = dl.get_test_df()

    model = KerasModel(x_train, y_train, x_valid, y_valid)
    model.train(IMG_TAR_SIZE)

    # get best threshold
    threshold_best = get_best_threshold(model, x_valid, y_valid)

    # get final result
    preds_test = model.predict(x_test, IMG_TAR_SIZE)
    pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('res\\' + SUBMISSION_NAME)


if __name__ == "__main__":
    main()
