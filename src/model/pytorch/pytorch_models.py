from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision.models.resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
from resnet_models import *
import torch.utils.model_zoo as model_zoo

"""
This script has been taken (and modified) from :
https://github.com/ternaus/TernausNet
@ARTICLE{arXiv:1801.05746,
         author = {V. Iglovikov and A. Shvets},
          title = {TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation},
        journal = {ArXiv e-prints},
         eprint = {1801.05746}, 
           year = 2018
        }
"""


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups),
                                  nn.BatchNorm2d(out_channels),
                                  )

    def forward(self, x):
        return self.conv(x)


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class CSE(nn.Module):
    def __init__(self, in_ch, r=2):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        input_x = x

        x = self.avg_pool(x).view(batch_size, channel_num)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(x, input_x)

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(x, input_x)

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=2):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        x = torch.add(cSE, sSE)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, groups=1):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, middle_channels, kernel_size=3, padding=1, groups=groups)
        self.conv2 = ConvBn2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.SCSE = SCSE(out_channels)

    def forward(self, x, e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        if e is not None:
            x = torch.cat([x, e], 1)
            x = F.dropout2d(x, p=0.50)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.SCSE(x)

        return x


class UNetResNet34_DS(nn.Module):

    def __init__(self, dropout_2d=0.2, pretrained=True):
        super().__init__()
        self.dropout_2d = dropout_2d

        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.resnet.layer1,
        )
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64 + 64, 32, 64)

        self.fuse_pixel = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.fuse_image = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.logit_image = nn.Linear(64, 2)

        self.logit = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x - mean[2]) / std[2],
            (x - mean[1]) / std[1],
            (x - mean[0]) / std[0],
        ], 1)
        """
        if INPUT_CHANNEL == 1:
            x = torch.cat([x,x,x],1)
        """

        e1 = self.encoder1(x)  # ;print('e1', e1.size())
        e2 = self.encoder2(e1)  # ;print('e2', e2.size())
        e3 = self.encoder3(e2)  # ;print('e3', e3.size())
        e4 = self.encoder4(e3)  # ;print('e4', e4.size())
        e5 = self.encoder5(e4)  # ;print('e5', e5.size())

        f = self.center(e5)  # ;print('f', f.size())

        d5 = self.decoder5(f, e5)  # ;print('d5', d5.size())
        d4 = self.decoder4(d5, e4)  # ;print('d4', d4.size())
        d3 = self.decoder3(d4, e3)  # ;print('d3', d3.size())
        d2 = self.decoder2(d3, e2)  # ;print('d2', d2.size())
        d1 = self.decoder1(d2, e1)  # ;print('d1', d1.size())

        # hyper column
        d = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        d = F.dropout2d(d, p=self.dropout_2d)
        fuse_pixel = self.fuse_pixel(d)
        logit_pixel = self.logit_pixel(fuse_pixel)

        e = F.adaptive_avg_pool2d(e5, output_size=[1, 1])
        e = F.dropout(e, p=self.dropout_2d)
        fuse_image = self.fuse_image(e)
        fuse_image_flatten = fuse_image.view(fuse_image.size(0), -1)
        logit_image = self.logit_image(fuse_image_flatten)

        logit = self.logit(torch.cat([
            fuse_pixel,
            F.interpolate(fuse_image.view(batch_size, -1, 1, 1, ), scale_factor=128, mode='nearest')], 1))

        return logit, logit_pixel, logit_image


class UNetResNext50_DS(nn.Module):
    def __init__(self, dropout_2d=0.2):
        super(UNetResNext50_DS, self).__init__()
        self.dropout_2d = dropout_2d
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.encoder1 = nn.Sequential(self.encoder.layer0.conv1,
                                      self.encoder.layer0.bn1,
                                      self.encoder.layer0.relu1)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1, groups=4),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.decoder5 = Decoder(256 + 512 * 4, 512, 64)
        self.decoder4 = Decoder(64 + 256 * 4, 256, 64)
        self.decoder3 = Decoder(64 + 128 * 4, 128, 64)
        self.decoder2 = Decoder(64 + 64 * 4, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.fuse_pixel = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.fuse_image = nn.Sequential(
            nn.Conv2d(512 * 4, 64, kernel_size=1, groups=4),
            nn.ReLU(inplace=True),
        )

        self.logit_image = nn.Linear(64, 2)

        self.logit = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        '''
        mean= [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0],
        ],1)
        '''
        if INPUT_CHANNEL == 1:
            x = torch.cat([x, x, x], 1)

        e1 = self.encoder1(x)  # ;print('e1', e1.size())
        e2 = self.encoder2(e1)  # ;print('e2', e2.size())
        e3 = self.encoder3(e2)  # ;print('e3', e3.size())
        e4 = self.encoder4(e3)  # ;print('e4', e4.size())
        e5 = self.encoder5(e4)  # ;print('e5', e5.size())

        f = self.center(e5)  # ;print('f', f.size())

        d5 = self.decoder5(f, e5)  # ;print('d5', d5.size())
        d4 = self.decoder4(d5, e4)  # ;print('d4', d4.size())
        d3 = self.decoder3(d4, e3)  # ;print('d3', d3.size())
        d2 = self.decoder2(d3, e2)  # ;print('d2', d2.size())
        d1 = self.decoder1(d2)  # ;print('d1', d1.size())

        # hyper column
        d = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        d = F.dropout2d(d, p=self.dropout_2d)
        fuse_pixel = self.fuse_pixel(d)
        logit_pixel = self.logit_pixel(fuse_pixel)

        e = F.adaptive_avg_pool2d(e5, output_size=[1, 1])
        e = F.dropout(e, p=self.dropout_2d)
        fuse_image = self.fuse_image(e)
        fuse_image_flatten = fuse_image.view(fuse_image.size(0), -1)
        logit_image = self.logit_image(fuse_image_flatten)

        logit = self.logit(torch.cat([
            fuse_pixel,
            F.interpolate(fuse_image.view(batch_size, -1, 1, 1, ), scale_factor=128, mode='nearest')], 1))

        return logit, logit_pixel, logit_image