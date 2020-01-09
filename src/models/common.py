import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, bn=False, act=False, **extra)
        self._seq = nn.Sequential()
        self._seq.add_module('_conv', nn.Conv2d(
            in_ch, out_ch, kernel, 
            **extra
        ))
        if bn:
            self._seq.add_module('_bn', nn.BatchNorm2d(out_ch))
        if act:
            self._seq.add_module('_act', relu())

    def forward(self, x):
        return self._seq(x)


class SameConv(BasicConv):
    def __init__(self, in_ch, out_ch, kernel, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, kernel, bn, act, stride=1, padding=kernel//2, **extra)


class SameConv3x3(SameConv):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, 3, bn, act, **extra)


class SameConv5x5(SameConv):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, 5, bn, act, **extra)


class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, 1, bn, act, **extra)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, 3, bn, act, **extra)


class Conv5x5(BasicConv):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **extra):
        super().__init__(in_ch, out_ch, 5, bn, act, **extra)


class MaxPool2x2(nn.MaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0))