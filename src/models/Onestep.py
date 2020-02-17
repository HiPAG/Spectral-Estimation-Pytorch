import torch
import torch.nn as nn

from .common import Conv1x1, Conv3x3, Conv5x5, BasicConv, SameConv, SameConv3x3, SameConv5x5


__ENTRANCE__ = 'Onestep'


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

class SA3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernal = 3, rate = 1, res = True):
        super().__init__()

        self.AS2 = nn.Conv3d(in_ch, out_ch, (1,kernal,kernal), dilation=rate, padding=(0,((kernal-1)//2)*rate,((kernal-1)//2)*rate))
        self.AS1 = nn.Conv3d(out_ch, out_ch, (kernal,1,1), dilation=rate, padding=(((kernal-1)//2)*rate,0,0))

        self.BS2 = nn.Conv3d(in_ch, out_ch, (1,kernal,kernal), dilation=rate, padding=(0,((kernal-1)//2)*rate,((kernal-1)//2)*rate))
        
        self.SC = nn.Conv3d(out_ch*2, out_ch,(1,1,1))
        
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.res = res

    def forward(self, x):
        ax2 = self.AS2(x)
        ax1 = self.AS1(ax2)

        bx2 = self.BS2(x)
        
        result = self.relu(self.SC(torch.cat([ax1,bx2],1)))

        if self.res:
            result = result + x

        return result

class Res_3DBlock(nn.Module):
    def __init__(self, n_chns, kernal_size=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            SA3D(n_chns, n_chns),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            SA3D(n_chns, n_chns),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.conv2(self.conv1(x))

class Onestep(nn.Module):
    def __init__(self, n_in, n_out, n_resblocks):
        super().__init__()
        deconv_width = 7
        self.cut = n_resblocks*2 + 2*2 - int(deconv_width/2)
        assert self.cut > 0

        # FSRCNN notation. You can take a look at FSRCNN paper for a deeper explanation
        s, d, m = (128, 32, n_resblocks)
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

        self.expand3D = nn.Sequential(
                            SA3D(1, d),
                            nn.PReLU()
                        )
        self.body3D = nn.Sequential(
            *[Res_3DBlock(d) for _ in range(m)]
        )
        self.ReOut = nn.Sequential(
            nn.Conv3d(d, 1, 1),
            # nn.Sigmoid()
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
        x = self.conv_out(x) + self.upsample(x_)
        
        # 3D net
        x = torch.unsqueeze(x, 1)
        # x = x.unsequeeze(1)
        x = self.expand3D(x)
        x = self.body3D(x)
        result = self.ReOut(x)

        return torch.squeeze(result, dim=1)
        # return result.sequeeze(1)
        # return self.conv_out(x) + self.upsample(x_)

 
if __name__ == '__main__':
    temp = torch.Tensor(2, 3, 256, 256).cuda()
    model = Onestep(3, 31, 2).cuda()
    output = model(temp)
    print(output.shape)
