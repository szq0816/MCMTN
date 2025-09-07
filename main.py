
import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange

from Local_block import SS_PW_Local_Conv
from MCM import MCM
from Norm_attn import Attention


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.norm(x)
        return out


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


def conv_3x3_bn(inp, oup, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class MCMTN(nn.Module):
    def __init__(self, dataset, image_size, in_channels, num_classes, K = 3,  ablation = 0):
        super().__init__()
        self.ih, self.iw = image_size
        self.ablation = ablation

        if dataset == 'IP' or 'HUST2013' or 'WH_LK':
            channels = [32, 64, 128]
        elif dataset == 'Loukia':
            channels = [16, 36, 64]

        self.stem = conv_3x3_bn(in_channels, channels[0])

        self.sqrt_lastchannle = int(math.sqrt(channels[-1]))

        self.s1 = SS_PW_Local_Conv(channels[0], channels[1], image_size, ablation)
        self.s2 = SS_PW_Local_Conv(channels[1], channels[2], image_size, ablation)


        if ablation != 3:
            self.g1 = MCM(channels[2], K)
            self.g2 = MCM(channels[2], K)
        else:
            print('消融')
            self.g1 = nn.Sequential(
                Rearrange('b c ih iw -> b (ih iw) c'),
                PreNorm(channels[2]),
                Attention(channels[2]),
                Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            )

        if ablation == 2:
            print('进行全局模块的消融')

        self.pool = GlobalAvgPool2d()
        self.fc = nn.Linear(channels[2], num_classes, bias=False)

    def forward(self, x):

        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        if self.ablation != 2:
            x = self.g1(x)
            # x = self.g2(x)
            # x = self.g3(x)

        # print(x.shape)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = MCMTN(dataset='HUST2013', image_size=(15, 15), in_channels=144, num_classes=15, K=7, ablation=0)
    net.eval()
    # print(net)
    input = torch.randn(64, 144, 15, 15)
    y = net(input)
    print(y.shape, count_parameters(net))

