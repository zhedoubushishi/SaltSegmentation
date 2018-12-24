# encoding=utf8
import cv2
import shutil
import numpy as np
import torch
from torch import nn
from datetime import datetime
from functools import reduce

################################################################################
# related functions & loss functions
################################################################################


def upsample(img):
    if IMG_ORI_SIZE == IMG_TAR_SIZE:
        return img
    return cv2.resize(img, (IMG_TAR_SIZE, IMG_TAR_SIZE))


def downsample(img):
    if IMG_ORI_SIZE == IMG_TAR_SIZE:
        return img
    return cv2.resize(img, (IMG_ORI_SIZE, IMG_ORI_SIZE))


def add_depth_channels(image_array, depth):
    image_array[:,:,1] = depth
    image_array[:,:,2] = image_array[:,:,0] * image_array[:,:,1]
    return image_array


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

        
def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True, cls=MyEncoder))
    log.write('\n')
    log.flush()

    
def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()
    return x.data.numpy()


def get_true_target(targets):
    truth_image = targets.squeeze(3).sum(2).sum(1) > 0
    return truth_image
  

def get_logits_outputs(outputs_image, outputs_pixel):
    batch_size, C, H, W = outputs_pixel.shape
    zero_mask = torch.zeros([batch_size, C, H, W], dtype=torch.float).to(device)
    empty_label = outputs_image<0
    outputs_pixel[empty_label] = zero_mask[empty_label]
    return outputs_pixel


def iou_numpy(outputs, labels):
    SMOOTH = 1e-6
    labels = labels.squeeze(1)
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded.mean()


def my_iou_metric(label, pred):
    return iou_numpy(pred > 0.5, label>0.5)


def my_iou_metric_2(label, pred):
    return iou_numpy(pred > 0, label>0.5)


def my_iou_metric_pad(label, pred):
    pad_size = (IMG_TAR_SIZE-SCALE*IMG_ORI_SIZE)//2
    return iou_numpy(pred[:,:,pad_size:-pad_size-1,pad_size+1:-pad_size]>0,label[:,:,pad_size:-pad_size-1,pad_size+1:-pad_size]>0.5)

    
def save_checkpoint(state, is_best, filename):
    check_filename = 'checkpoint_{}_resize.pth.tar'.format(filename)
    torch.save(state, check_filename)
    if is_best:
        shutil.copyfile(check_filename, 'model_best_{}_resize.pth.tar'.format(filename))
        

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
