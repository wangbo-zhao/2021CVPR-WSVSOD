from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_dilation import resnet50, Bottleneck, conv1x1


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class edge_refine_aspp(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(edge_refine_aspp, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

        self.fuse = nn.Conv2d(reduction_dim * 5, reduction_dim, kernel_size=1, bias=False)
    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)

        out = self.fuse(out)


        return out




class RefinementModule(nn.Module):
    """ Reduce channels and refinment module"""

    def __init__(self,
        bottom_up_channels,
        reduce_channels,
        top_down_channels,
        refinement_channels,
        expansion=2
    ):
        super(RefinementModule, self).__init__()
        downsample = None
        if bottom_up_channels != reduce_channels:
            downsample = nn.Sequential(
                conv1x1(bottom_up_channels, reduce_channels),
                nn.BatchNorm2d(reduce_channels),
            )
        self.skip = Bottleneck(bottom_up_channels, reduce_channels // expansion, 1, 1, downsample, expansion)
        self.refine = _ConvBatchNormReLU(reduce_channels + top_down_channels, refinement_channels, 3, 1, 1, 1)
    def forward(self, td, bu):

        td = self.skip(td)
        x = torch.cat((bu, td), dim=1)
        x = self.refine(x)

        return x