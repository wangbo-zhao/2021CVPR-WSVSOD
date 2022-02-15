from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import edge_refine_aspp, RefinementModule
from .ConvLSTM import ConvLSTMCell
from .resnet_dilation import resnet50
from .edge import Edge_Module, RCAB
import numpy as np
import cv2



class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, relu=True):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if relu == True:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x

class VideoEncoderPart(nn.Module):

    def __init__(self, output_stride, input_channels=3, pretrained=False):
        super(VideoEncoderPart, self).__init__()

        self.resnet = resnet50(pretrained=pretrained, output_stride=output_stride, input_channels=input_channels)
        self.edge_layer = Edge_Module(in_fea=[256, 512, 1024], mid_fea=32)
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.aspp = edge_refine_aspp(2048, 128, output_stride)


        if pretrained:
            for key in self.state_dict():
                if 'resnet' not in key:
                    self.init_layer(key)

    def init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001


    def forward(self, x):

        block0 = self.resnet.conv1(x)
        block0 = self.resnet.bn1(block0)
        block0 = self.resnet.relu(block0)
        block0 = self.resnet.maxpool(block0)

        block1 = self.resnet.layer1(block0)
        block2 = self.resnet.layer2(block1)
        block3 = self.resnet.layer3(block2)
        block4 = self.resnet.layer4(block3)

        edge_map = self.edge_layer(block1, block2, block3)

        block4 = self.aspp(block4)

        return block1, block2, block3, block4, edge_map


class VideoEncoder(nn.Module):

    def __init__(self, output_stride, input_channels=3, pretrained=True):
        super(VideoEncoder, self).__init__()
        self.encoder = VideoEncoderPart(output_stride, input_channels=3, pretrained=True)

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                print(m[0], "eval")
                m[1].eval()

    def forward(self, rgb, flow):
        rgb_block1, rgb_block2, rgb_block3, rgb_block4, rgb_edge_map  = self.encoder(rgb)
        flow_block1, flow_block2, flow_block3, flow_block4, flow_edge_map = self.encoder(flow)

        rgb_features = [rgb_block1, rgb_block2, rgb_block3, rgb_block4, rgb_edge_map]
        flow_features = [flow_block1, flow_block2, flow_block3, flow_block4, flow_edge_map]
        return rgb_features, flow_features

class Temporal(nn.Module):

    def __init__(self, in_channels):
        super(Temporal, self).__init__()
        self.forward_lstm = ConvLSTMCell(input_size=in_channels, hidden_size=in_channels)
        self.backward_lstm = ConvLSTMCell(input_size=in_channels, hidden_size=in_channels)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels= 2 * in_channels, out_channels=in_channels, stride=1, kernel_size=3, padding=1),
                                  nn.ReLU())


    def forward(self, feats_time):
#forward
        state = None
        feats_forward = []
        for i in range(feats_time.shape[2]):
            state = self.forward_lstm(feats_time[:, :, i, :, :], state)
            feats_forward.append(state[0])

#backward
        state = None
        feats_backward = []
        for i in range(feats_time.shape[2]):
            state = self.backward_lstm(feats_forward[len(feats_forward) -1 - i], state)
            feats_backward.append(state[0])

        feats_backward = feats_backward[::-1]

#fuse
        feats = []
        for i in range(feats_time.shape[2]):
            feat = torch.tanh(self.fuse(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)

        feats = torch.stack(feats, dim=2)
        return feats




class RGBFlowFusion(nn.Module):
    def __init__(self):
        super(RGBFlowFusion, self).__init__()
        self.fuse = nn.Conv2d(in_channels=64, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.gate = nn .Conv2d(in_channels=32, out_channels=2, stride=1, kernel_size=3, padding=1)

        self.mlp_rgb = nn.Linear(32, 32)
        self.mlp_flow = nn.Linear(32, 32)

        self.conv_rgb = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_flow = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb_fea, flow_fea):
        rgb_flow_fea = self.fuse(torch.cat((rgb_fea, flow_fea), dim=1))

        gate = F.adaptive_avg_pool2d(torch.sigmoid(self.gate(rgb_flow_fea)), (1, 1))
        gate_rgb = gate[:, :1, :, :]
        gate_flow = gate[:, 1:, :, :]

        rgb_fea = gate_rgb * rgb_fea
        flow_fea = gate_flow * flow_fea

        N, C, h, w = rgb_flow_fea.shape

        attention_feature_max = F.adaptive_max_pool2d(rgb_flow_fea, (1, 1)).view(N, C)

        channel_attention_rgb = F.softmax(self.mlp_rgb(attention_feature_max), dim=1).unsqueeze(2).unsqueeze(3)
        channel_attention_flow = F.softmax(self.mlp_flow(attention_feature_max), dim=1).unsqueeze(2).unsqueeze(3)


        spatial_attention_rgb = torch.sigmoid(self.conv_rgb(rgb_flow_fea))
        spatial_attention_flow = torch.sigmoid(self.conv_flow(rgb_flow_fea))

        fuse = rgb_fea + rgb_fea * channel_attention_rgb * spatial_attention_rgb  + flow_fea + flow_fea * channel_attention_flow * spatial_attention_flow

        return fuse



class SalientContextPrior(nn.Module):

    def __init__(self):
        super(SalientContextPrior, self).__init__()
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, stride=1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, stride=1, kernel_size=1),
        )
        self.inter_channels = 8



    def forward(self, video_features):
        batch_size = video_features.shape[0]
        x1 = self.g(video_features[:, :, 0, :, :]).view(batch_size, self.inter_channels, -1)
        x2 = self.g(video_features[:, :, 1, :, :]).view(batch_size, self.inter_channels, -1)
        x3 = self.g(video_features[:, :, 2, :, :]).view(batch_size, self.inter_channels, -1)
        x4 = self.g(video_features[:, :, 3, :, :]).view(batch_size, self.inter_channels, -1)


        x11 = torch.sigmoid(torch.matmul(x1.permute(0, 2, 1), x1))
        x12 = torch.sigmoid(torch.matmul(x1.permute(0, 2, 1), x2))
        x13 = torch.sigmoid(torch.matmul(x1.permute(0, 2, 1), x3))
        x14 = torch.sigmoid(torch.matmul(x1.permute(0, 2, 1), x4))

        x22 = torch.sigmoid(torch.matmul(x2.permute(0, 2, 1), x2))
        x23 = torch.sigmoid(torch.matmul(x2.permute(0, 2, 1), x3))
        x24 = torch.sigmoid(torch.matmul(x2.permute(0, 2, 1), x4))

        x33 = torch.sigmoid(torch.matmul(x3.permute(0, 2, 1), x3))
        x34 = torch.sigmoid(torch.matmul(x3.permute(0, 2, 1), x4))

        x44 = torch.sigmoid(torch.matmul(x4.permute(0, 2, 1), x4))

        pred_prior = torch.cat((x11, x12, x13, x14, x22, x23, x24, x33, x34, x44), dim=0)

        return pred_prior




class VideoDecoder(nn.Module):

    def __init__(self):
        super(VideoDecoder, self).__init__()

        self.conv1rgb = nn.Conv2d(256, 32, kernel_size=1, stride=1)
        self.conv2rgb = nn.Conv2d(512, 32,kernel_size=1, stride=1)
        self.conv3rgb = nn.Conv2d(1024,32, kernel_size=1, stride=1)
        self.conv4rgb = nn.Conv2d(128, 32, kernel_size=1, stride=1)

        self.conv1flow = nn.Conv2d(256, 32, kernel_size=1, stride=1)
        self.conv2flow = nn.Conv2d(512, 32, kernel_size=1, stride=1)
        self.conv3flow = nn.Conv2d(1024, 32, kernel_size=1, stride=1)
        self.conv4flow = nn.Conv2d(128, 32, kernel_size=1, stride=1)

        self.rgbflowfusion1 = RGBFlowFusion()
        self.rgbflowfusion2 = RGBFlowFusion()
        self.rgbflowfusion3 = RGBFlowFusion()
        self.rgbflowfusion4 = RGBFlowFusion()

        self.temporal1 = Temporal(in_channels=32)
        self.temporal2 = Temporal(in_channels=32)
        self.temporal3 = Temporal(in_channels=32)
        self.temporal4 = Temporal(in_channels=32)

        self.conv43 = ConvBnRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv432 = ConvBnRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv4321 = ConvBnRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, relu=True)

        self.pred = nn.Conv2d(32, 1, kernel_size=1, stride=1)


        self.prior = SalientContextPrior()

        self.sal_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(64)
        self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, rgb_features, flow_features):

        rgb_flow_x1s = []
        rgb_flow_x2s = []
        rgb_flow_x3s = []
        rgb_flow_x4s = []


        for i in range(len(rgb_features)):
            rgb_feature = rgb_features[i]
            flow_feature = flow_features[i]


            rgb_block1, rgb_block2, rgb_block3, rgb_block4, rgb_edge_map = rgb_feature
            flow_block1, flow_block2, flow_block3, flow_block4, flow_edge_map = flow_feature

            rgb_x1 = F.relu(self.conv1rgb(rgb_block1))
            rgb_x2 = F.relu(self.conv2rgb(rgb_block2))
            rgb_x3 = F.relu(self.conv3rgb(rgb_block3))
            rgb_x4 = F.relu(self.conv4rgb(rgb_block4))

            flow_x1 = F.relu(self.conv1flow(flow_block1))
            flow_x2 = F.relu(self.conv2flow(flow_block2))
            flow_x3 = F.relu(self.conv3flow(flow_block3))
            flow_x4 = F.relu(self.conv4flow(flow_block4))


            rgb_flow_x1 = self.rgbflowfusion1(rgb_x1, flow_x1)
            rgb_flow_x1s.append(rgb_flow_x1)

            rgb_flow_x2 = self.rgbflowfusion2(rgb_x2, flow_x2)
            rgb_flow_x2s.append(rgb_flow_x2)

            rgb_flow_x3 = self.rgbflowfusion3(rgb_x3, flow_x3)
            rgb_flow_x3s.append(rgb_flow_x3)

            rgb_flow_x4 = self.rgbflowfusion4(rgb_x4, flow_x4)
            rgb_flow_x4s.append(rgb_flow_x4)


        rgb_flow_x1s = torch.stack(rgb_flow_x1s, dim=2)
        rgb_flow_temporal_x1s = self.temporal1(rgb_flow_x1s)

        rgb_flow_x2s = torch.stack(rgb_flow_x2s, dim=2)
        rgb_flow_temporal_x2s = self.temporal2(rgb_flow_x2s)

        rgb_flow_x3s = torch.stack(rgb_flow_x3s, dim=2)
        rgb_flow_temporal_x3s = self.temporal3(rgb_flow_x3s)

        rgb_flow_x4s = torch.stack(rgb_flow_x4s, dim=2)
        rgb_flow_temporal_x4s = self.temporal4(rgb_flow_x4s)

        sal_fuse_inits, sal_refs = [], []
        for_prior = []
        for i in range(rgb_flow_x1s.shape[2]):
            rgb_edge_map = rgb_features[i][-1]

            rgb_flow_temporal_x1 = rgb_flow_temporal_x1s[:, :, i, :, :]
            rgb_flow_temporal_x2 = rgb_flow_temporal_x2s[:, :, i, :, :]
            rgb_flow_temporal_x3 = rgb_flow_temporal_x3s[:, :, i, :, :]
            rgb_flow_temporal_x4 = rgb_flow_temporal_x4s[:, :, i, :, :]

            conv43 = self.conv43(torch.cat((rgb_flow_temporal_x4, rgb_flow_temporal_x3), dim=1))
            conv43 = F.upsample(conv43, scale_factor=2, mode='bilinear', align_corners=True)


            conv432 = self.conv432(torch.cat((conv43, rgb_flow_temporal_x2), dim=1))
            conv432 = F.upsample(conv432, scale_factor=2, mode='bilinear', align_corners=True)

            conv4321 = self.conv4321(torch.cat((conv432, rgb_flow_temporal_x1), dim=1))
            for_prior.append(conv4321)

            fuse_feature = F.upsample(conv4321, scale_factor=4, mode='bilinear')

            sal_fuse_init = self.pred(fuse_feature)

            sal_feature = F.relu(self.sal_conv(sal_fuse_init))
            edge_feature = F.relu(self.edge_conv(rgb_edge_map))
            sal_edge_feature = torch.cat((sal_feature, edge_feature), dim=1)
            sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
            sal_ref = self.fused_edge_sal(sal_edge_feature)

            sal_fuse_inits.append(sal_fuse_init)
            sal_refs.append(sal_ref)
        #
        return sal_fuse_inits, sal_refs
        # for_prior = torch.stack(for_prior, dim=2)
        # pred_prior = self.prior(for_prior)
        #
        # return sal_fuse_inits, sal_refs, pred_prior
