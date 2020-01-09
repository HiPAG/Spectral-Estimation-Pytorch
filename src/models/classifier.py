import torch
import torch.nn as nn

from .common import SameConv3x3, SameConv5x5, MaxPool2x2


__ENTRANCE__ = 'Classifier'


class Classifier(nn.Module):
    def __init__(n_in, n_out):
        super().__init__()

        self.conv1 = SameConv5x5(n_in, 256, act=True)
        self.pool1 = MaxPool2x2()

        self.conv2 = SameConv3x3(256, 128, act=True)
        self.pool2 = MaxPool2x2()

        self.conv3 = SameConv3x3(128, 64, act=True)
        self.pool3 = MaxPool2x2()

        self.conv4 = SameConv3x3(64, n_out)
        self.pool4 = MaxPool2x2()

        self.act_out = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        y = self.pool4(x)

        return self.act_out(y.view(*y.size()[:2]))
