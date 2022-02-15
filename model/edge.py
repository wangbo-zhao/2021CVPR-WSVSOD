
import torch
import torch.nn as nn
import torch.nn.functional as F

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res



class Edge_Module(nn.Module):

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=32):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_fea[0], mid_fea, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(mid_fea, mid_fea, 3, padding=1),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_fea[1], mid_fea, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(mid_fea, mid_fea, 3, padding=1),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_fea[2], mid_fea, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(mid_fea, mid_fea, 3, padding=1),
                                   nn.ReLU())


        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        x1_fea = self.conv1(x1)
        x2_fea = self.conv2(x2)
        x3_fea = self.conv3(x3)

        edge1 = F.interpolate(x1_fea, scale_factor=4, mode='bilinear', align_corners=True)
        edge2 = F.interpolate(x2_fea, scale_factor=8, mode='bilinear', align_corners=True)
        edge3 = F.interpolate(x3_fea, scale_factor=16, mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge


