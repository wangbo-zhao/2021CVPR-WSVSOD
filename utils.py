
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import cv2

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5
contour_th_img = 1.0

def adjust_lr(optimizer, epoch, decay_rate=0.9, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    print("lr=", optimizer.param_groups[0]["lr"])

    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad




def visualize_prediction(pred, save_path, prediction_name):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_{}.png'.format(kk, prediction_name)
        cv2.imwrite(save_path + name, pred_edge_kk)



