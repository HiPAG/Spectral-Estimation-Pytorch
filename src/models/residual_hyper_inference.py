import torch
import torch.nn as nn

from .common import Conv1x1, Conv3x3, Conv5x5, BasicConv, SameConv, SameConv3x3, SameConv5x5


__ENTRANCE__ = 'ResidualHyperInference'


class ResBlock(nn.Module):
    def __init__(self, n_chns):
        super().__init__()
        self.conv1 = nn.Sequential(
            SameConv3x3(n_chns, n_chns),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            SameConv3x3(n_chns, n_chns),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.conv2(self.conv1(x))


class ResidualHyperInference(nn.Module):
    def __init__(self, n_in, n_out, n_resblocks):
        super().__init__()
        deconv_width = 7
        self.cut = n_resblocks*2 + 2*2 - int(deconv_width/2)
        assert self.cut > 0

        # FSRCNN notation. You can take a look at FSRCNN paper for a deeper explanation
        s, d, m = (128, 32, n_resblocks)

        self.temp = 0

        self.conv_feature = nn.Sequential(
            SameConv5x5(n_in, s),
            nn.PReLU()
        )

        self.conv_shrink = nn.Sequential(
            Conv1x1(s, d),
            nn.PReLU()
        )

        self.body = nn.Sequential(
            *[ResBlock(d) for _ in range(m)]
        )

        self.conv_expand = nn.Sequential(
            Conv1x1(d, s),
            nn.PReLU()
        )

        self.upsample = SameConv(n_in, n_out, deconv_width)

        self.conv_out = nn.Sequential(
            SameConv5x5(s, n_out),
            nn.PReLU()
        )

    def forward(self, x):
        x_ = x
        # To feature space
        x = self.conv_feature(x)

        x1 = x

        # Shrinking
        x = self.conv_shrink(x)

        x2 = x

        # Feature mapping
        x = self.body(x)

        x = x + x2

        # Expanding
        x = self.conv_expand(x)

        x = x + x1
        result = nn.functional.relu(self.conv_out(x) + self.upsample(x_))

        return result
        # return self.conv_out(x) + self.upsample(x_)


if __name__ == '__main__':
    temp = torch.Tensor(2, 96, 256, 256)
    model = ResidualHyperInference(96, 31, 2)
    output = model(temp)
    print(output.shape)
