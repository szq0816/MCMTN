'''
这里是局部模块的改进
改成1*1空谱卷积，
'''

import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

class Asym_conv_Block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Asym_conv_Block, self).__init__()
        self.conv_1x1 = nn.Sequential( nn.Conv2d(in_channels, out_channels, 1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())

        self.asym_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1),
                                      padding=(1, 0), bias=False)
        self.asym_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3),
                                      padding=(0, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.asym_conv1(out)
        out = self.asym_conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

class Tranpose_PW_Conv(nn.Module):
    def __init__(self, in_channels, image_size):
        super(Tranpose_PW_Conv, self).__init__()
        self.h, self.w = image_size

        self.to_channel_last = nn.Sequential(Rearrange('b c h w -> b (h w) c'),
                                            nn.LayerNorm(in_channels))

        self.conv_1x1 = nn.Sequential( nn.Conv2d(self.h * self.w, self.h * self.w, 1),
                                        nn.BatchNorm2d(self.h * self.w),
                                        nn.ReLU())

        self.to_channel_first = Rearrange('b (h w) c -> b c h w', h=self.h, w=self.w)

    def forward(self, x):
        identity = x
        x = self.to_channel_last(x).unsqueeze(-1)
        x = self.conv_1x1(x)
        x = self.to_channel_first(x.squeeze(-1))
        out = x + identity
        return out


class SS_PW_Local_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, ablation, expansion = 2):
        super(SS_PW_Local_Conv, self).__init__()
        self.ablation = ablation
        self.inner_channels = in_channels * expansion
        #1+3+1+tranpose1的形式
        #通道上升
        self.conv_1x1_in = nn.Sequential( nn.Conv2d(in_channels, self.inner_channels, 1),
                                        nn.BatchNorm2d(self.inner_channels),
                                        nn.ReLU())

        #通道保持,进行局部信息的提取
        self.cnv_PW_3x3 = nn.Sequential(nn.Conv2d(self.inner_channels,self.inner_channels,
                                               kernel_size=3, stride=1, padding=1, groups=self.inner_channels),
                                        nn.BatchNorm2d(self.inner_channels) )

        #通道下降
        self.conv_1x1_out = nn.Sequential( nn.Conv2d(self.inner_channels, out_channels, 1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())

        # Transpose的1*1
        if ablation!=1:
            self.spatial_Global_conv = Tranpose_PW_Conv(out_channels, image_size)
            # self.conv_huan = nn.Conv2d(out_channels, out_channels, 1)
        else:
            print('进行消融')
        self.down_sample = nn.Sequential( nn.Conv2d(in_channels, out_channels, 1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU())

    def forward(self,x):
        identity = x
        x = self.conv_1x1_in(x)
        x = self.cnv_PW_3x3(x)
        x = self.conv_1x1_out(x)
        if self.ablation!=1:
            x = self.spatial_Global_conv(x)
            # x = self.conv_huan(x)
        return x + self.down_sample(identity)

if __name__ == '__main__':


    model = SS_PW_Local_Conv(36, 64, (15, 15), 0)
    model.eval()
    print(model)
    input = torch.randn(64, 36, 15, 15)
    y = model(input)
    print(y.size())