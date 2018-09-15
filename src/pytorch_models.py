import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(_in, _out):
    return nn.Conv2d(_in, _out, 3, stride=(1,1), padding=1)


class BatchActivate(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.batch = nn.BatchNorm2d(_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.batch(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
