import torch
import torch.nn as nn

from .common import SameConv3x3, SameConv5x5, MaxPool2x2


__ENTRANCE__ = 'Estimator'


class Estimator(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()

        assert n_out % n_in == 0
        self.out_shape = (n_out//n_in, n_in)

        self.conv1 = SameConv5x5(n_in, 16, act=True)
        self.conv2 = SameConv5x5(16, 32, act=True)
        self.pool1 = MaxPool2x2()
        
        self.conv3 = SameConv5x5(32, 64, act=True)
        self.conv4 = SameConv5x5(64, 64, act=True)
        self.pool2 = MaxPool2x2()

        self.conv5 = SameConv3x3(64, 64, act=True)
        self.conv6 = SameConv3x3(64, 64, act=True)
        self.pool3 = MaxPool2x2()

        self.conv7 = SameConv3x3(64, 64, act=True)
        self.conv8 = SameConv3x3(64, 64, act=True)
        self.pool4 = MaxPool2x2()

        self.conv9 = SameConv3x3(64, 128, act=True)
        self.conv10 = SameConv3x3(128, 128, act=True)

        self.conv11 = SameConv3x3(128, 128, act=True)
        self.conv12 = SameConv3x3(128, n_out, act=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)

        x = self.conv9(x)
        x = self.conv10(x)

        x = self.conv11(x)
        y = self.conv12(x)

        c = y.size(1)
        y = y.mean(0).view(c, -1).mean(-1)
        return y.view(*self.out_shape)
