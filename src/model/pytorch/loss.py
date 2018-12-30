from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class Dice_Bce_Loss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7, dice_weight=0.5, 
                 dice_loss=None, bce_weight=0.9, bce_loss=None):
        super(Dice_Bce_Loss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = bce_loss
        self.dice_loss = dice_loss
        
        if self.bce_loss is None:
            self.bce_loss = F.binary_cross_entropy_with_logits
        if self.dice_loss is None:
            self.dice_loss = DiceLoss(smooth, eps)
            
        self.activation = torch.sigmoid
            
    def forward(self, output, target):
        output = self.activation(output)
        return self.dice_weight * self.dice_loss(output, target) + self.bce_weight * self.bce_loss(output, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted)+1, grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class BCE_Lovaz_Loss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        super(BCE_Lovaz_Loss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore
        
    def forward(self, logits, targets):
        loss_1 = lovasz_hinge(logits.squeeze(1), targets.squeeze(1), self.per_image, self.ignore)
        loss_2 = nn.BCEWithLogitsLoss()(logits, targets)
        loss = loss_1 + loss_2
        return loss
    
class Lovaz_Loss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        super(Lovaz_Loss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore
        
    def forward(self, logits, targets):
        loss = lovasz_hinge(logits.squeeze(1), targets.squeeze(1), self.per_image, self.ignore)
        return loss

class Fuse_Loss(nn.Module):
    def __init__(self, pixel_loss_func=Lovaz_Loss(), weight_image=0.05, weight_pixel=0.5, weight=1):
        super(Fuse_Loss, self).__init__()
        self.image_loss_func = F.cross_entropy
        self.pixel_loss_func = pixel_loss_func
        self.activation = torch.sigmoid
        self.weight_image = weight_image
        self.weight_pixel = weight_pixel
        self.weight = weight
        
    def forward(self, logits, logit_pixel, logit_image, truth_pixel, truth_image):
        pixel_non_empty = logit_pixel[truth_image,:,:,:]
        mask_non_empty = truth_pixel[truth_image,:,:,:]
        loss_pixel = self.pixel_loss_func(pixel_non_empty, mask_non_empty)
        truth_image = truth_image.type(torch.LongTensor).to(device)
        loss_image = self.image_loss_func(logit_image, truth_image, reduction='elementwise_mean')
        loss = self.pixel_loss_func(logits, truth_pixel)
        return self.weight_pixel*loss_pixel + self.weight_image*loss_image + self.weight*loss