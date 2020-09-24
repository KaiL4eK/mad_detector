import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from .common import Conv_Bn_Activation, YoloLayer, get_region_boxes, Upsample, ResConv2dBatchLeaky


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.conv1 = Conv_Bn_Activation(3, 32, 3, 2, 'leaky')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'leaky')
        self.conv3 = Conv_Bn_Activation(64, 64, 3, 1, 'leaky')

        self.res1 = ResConv2dBatchLeaky(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # , padding=2//2)
        self.conv4 = Conv_Bn_Activation(128, 128, 3, 1, 'leaky')

        self.res2 = ResConv2dBatchLeaky(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # , padding=2//2)
        self.conv5 = Conv_Bn_Activation(256, 256, 3, 1, 'leaky')

        self.res3 = ResConv2dBatchLeaky(256, True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # , padding=2//2)
        self.conv6 = Conv_Bn_Activation(512, 512, 3, 1, 'leaky')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        r1 = self.res1(x3)
        m1 = self.maxpool1(r1)
        x4 = self.conv4(m1)

        r2 = self.res2(x4)
        m2 = self.maxpool2(r2)
        x5 = self.conv5(m2)

        # r3 ~ x24
        r3, x23 = self.res3(x5)
        m3 = self.maxpool3(r3)
        x26 = self.conv6(m3)

        return x26, x23


class Yolov4Tiny(nn.Module):
    def __init__(self, config, inference=False):
        super(Yolov4Tiny, self).__init__()
        self.inference = inference

        n_classes = config['n_classes']
        strides = config['strides']
        anchors = config['anchors']
        anchor_masks = config['anchor_masks']

        output_ch = (4 + 1 + n_classes) * 3

        self.backbone = Backbone()

        # x16
        self.conv32 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.upsample1 = Upsample(inference)
        self.conv35 = Conv_Bn_Activation(384, 256, 3, 1, 'leaky')
        self.out_1 = Conv_Bn_Activation(
            256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.yolo1 = YoloLayer(
            anchor_mask=anchor_masks[0],
            num_classes=n_classes,
            anchors=anchors,
            stride=strides[0],
            inference=inference)

        # x32
        self.conv27 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv28 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.out_2 = Conv_Bn_Activation(
            512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.yolo2 = YoloLayer(
            anchor_mask=anchor_masks[1],
            num_classes=n_classes,
            anchors=anchors,
            stride=strides[1],
            inference=inference)

    def forward(self, input):
        # x<name> matches
        # https://lutzroeder.github.io/netron/?url=https%3A%2F%2Fraw.githubusercontent.com%2FAlexeyAB%2Fdarknet%2Fmaster%2Fcfg%2Fyolov4-tiny.cfg
        x26, x23 = self.backbone(input)

        # x32
        x27 = self.conv27(x26)

        x28 = self.conv28(x27)
        out2 = self.out_2(x28)
        y2 = self.yolo2(out2)

        # x16
        x32 = self.conv32(x27)
        up = self.upsample1(x32, x23.size())
        x34 = torch.cat([x23, up], dim=1)
        x35 = self.conv35(x34)
        out1 = self.out_1(x35)
        y1 = self.yolo1(out1)

        if self.inference:
            return get_region_boxes([y1, y2])
        else:
            return [y1, y2]
