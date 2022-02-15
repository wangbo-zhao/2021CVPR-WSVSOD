import os
import torch
import torch.nn as nn
from torch.utils import data
from datetime import datetime
import argparse
from dataset.data import VideoDataset
from dataset.transforms import get_train_transforms
from torch.utils.data import DataLoader
from model.model import VideoEncoder, VideoDecoder
import torch.optim as optim
from utils import adjust_lr, label_edge_prediction, visualize_prediction
from datetime import datetime
import smoothness
from torch.nn import functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=20, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=15, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
args = parser.parse_args()

transforms = get_train_transforms(input_size=(args.size, args.size))
dataset = VideoDataset(root_dir="the dir of your dataset", trainingset_list=["DAVIS", "DAVSOD"], trainging=True, transforms=transforms)


train_dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize, num_workers=6, shuffle=True, drop_last=True)

Encoder = VideoEncoder(output_stride=16, pretrained=True)
Decoder = VideoDecoder()

Encoder.load_state_dict(torch.load("./save_models/pretrained/scribble_Encoder_30.pth"))
Decoder_dict = Decoder.state_dict()
pretrained_dict = torch.load("./save_models/pretrained/scribble_Decoder_30.pth")

for k, v in pretrained_dict.items():
    if (k in Decoder_dict):
        print("load:%s" % k)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in Decoder_dict)}
Decoder_dict.update(pretrained_dict)
Decoder.load_state_dict(Decoder_dict)


Encoder = Encoder.cuda()
Decoder = Decoder.cuda()
print("network ready!!!")

optimizer_Encoder = optim.Adam(Encoder.parameters(), lr=args.lr)
optimizer_Decoder = optim.Adam(Decoder.parameters(), lr=args.lr)

CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)

def get_one_hot(mask, gt):

    a = torch.ones([1, 2, 4096]).cuda()
    a[:, 0, mask.view(4096) == 0] = 0
    a[:, 1, mask.view(4096) == 0] = 0

    a[:, 0, gt.view(4096) == 1] = 1
    a[:, 1, gt.view(4096) == 1] = 0

    a[:, 0, (mask - gt).view(4096) == 1] = 0
    a[:, 1, (mask - gt).view(4096) == 1] = 1

    return a

def prior_loss(prior, masks, gts):

    x1_mask = F.interpolate(masks[0], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x2_mask = F.interpolate(masks[1], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x3_mask = F.interpolate(masks[2], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x4_mask = F.interpolate(masks[3], scale_factor=0.25, mode="nearest").view(1, 1, -1)



    mask_11 = torch.matmul(x1_mask.permute(0, 2, 1), x1_mask)
    mask_12 = torch.matmul(x1_mask.permute(0, 2, 1), x2_mask)
    mask_13 = torch.matmul(x1_mask.permute(0, 2, 1), x3_mask)
    mask_14 = torch.matmul(x1_mask.permute(0, 2, 1), x4_mask)

    mask_22 = torch.matmul(x2_mask.permute(0, 2, 1), x2_mask)
    mask_23 = torch.matmul(x2_mask.permute(0, 2, 1), x3_mask)
    mask_24 = torch.matmul(x2_mask.permute(0, 2, 1), x4_mask)

    mask_33 = torch.matmul(x3_mask.permute(0, 2, 1), x3_mask)
    mask_34 = torch.matmul(x3_mask.permute(0, 2, 1), x4_mask)

    mask_44 = torch.matmul(x4_mask.permute(0, 2, 1), x4_mask)

    mask = torch.cat((mask_11, mask_12, mask_13, mask_14, mask_22, mask_23, mask_24, mask_33, mask_34, mask_44), dim=0)

    prior = mask * prior
    size = prior.shape[0] * prior.shape[1] * prior.shape[2]
    ratio = size / mask.sum()


    x1_gt = F.interpolate(gts[0], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x1_onehot = get_one_hot(x1_mask, x1_gt)

    x2_gt = F.interpolate(gts[1], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x2_onehot = get_one_hot(x2_mask, x2_gt)

    x3_gt = F.interpolate(gts[2], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x3_onehot = get_one_hot(x3_mask, x3_gt)

    x4_gt = F.interpolate(gts[3], scale_factor=0.25, mode="nearest").view(1, 1, -1)
    x4_onehot = get_one_hot(x4_mask, x4_gt)

    x11_sim = torch.matmul(x1_onehot.permute(0, 2, 1), x1_onehot)
    x12_sim = torch.matmul(x1_onehot.permute(0, 2, 1), x2_onehot)
    x13_sim = torch.matmul(x1_onehot.permute(0, 2, 1), x3_onehot)
    x14_sim = torch.matmul(x1_onehot.permute(0, 2, 1), x4_onehot)

    x22_sim = torch.matmul(x2_onehot.permute(0, 2, 1), x2_onehot)
    x23_sim = torch.matmul(x2_onehot.permute(0, 2, 1), x3_onehot)
    x24_sim = torch.matmul(x2_onehot.permute(0, 2, 1), x4_onehot)

    x33_sim = torch.matmul(x3_onehot.permute(0, 2, 1), x3_onehot)
    x34_sim = torch.matmul(x3_onehot.permute(0, 2, 1), x4_onehot)

    x44_sim = torch.matmul(x4_onehot.permute(0, 2, 1), x4_onehot)

    sim = torch.cat((x11_sim, x12_sim, x13_sim, x14_sim, x22_sim, x23_sim, x24_sim, x33_sim, x34_sim, x44_sim), dim=0)

    loss = ratio * CE(prior, sim)

    return loss


def train(train_loader, Encoder, Decoder, optimizer_Encoder, optimizer_Decoder, epoch):
    Encoder.train()
    Decoder.train()

    total_step = len(train_dataloader)
    for i, packs in enumerate(train_dataloader):
        optimizer_Encoder.zero_grad()
        optimizer_Decoder.zero_grad()

        loss = 0
        rgb_features = []
        flow_features = []
        edge_maps = []
        masks = []
        gts = []
        for pack in packs:
            images, flow = pack["image"], pack["flow"]
            images, flow = images.cuda(), flow.cuda()

            rgb_feature, flow_feature = Encoder(images, flow)
            edge_maps.append(rgb_feature[-1])
            rgb_features.append(rgb_feature)
            flow_features.append(flow_feature)

            masks.append(pack["mask"].cuda())
            gts.append(pack["gt"].cuda())


        sal1s, sal2s, pred_prior = Decoder(rgb_features, flow_features)

        pri_loss = prior_loss(pred_prior, masks, gts)
        loss = 0

        for k in range(len(sal1s)):
            sal1 = sal1s[k]
            sal2 = sal2s[k]
            mask = packs[k]["mask"].cuda()
            gt = packs[k]["gt"].cuda()
            grey = packs[k]["grey"].cuda()
            edge_map = edge_maps[k]


            img_size = sal1.size(2) * sal1.size(3) * sal1.size(0)
            ratio = img_size / torch.sum(mask)

            sal1_prob = torch.sigmoid(sal1)
            sal1_prob = sal1_prob * mask
            sal2_prob = torch.sigmoid(sal2)
            sal2_prob = sal2_prob * mask

            edges_gt = label_edge_prediction(torch.sigmoid(sal2).detach())
            edge_loss = CE(torch.sigmoid(edge_map), edges_gt)

            sal_loss1 = ratio * CE(sal1_prob, gt*mask) + 0.3 * smooth_loss(torch.sigmoid(sal1), grey)
            sal_loss2 = ratio * CE(sal2_prob, gt*mask) + 0.3 * smooth_loss(torch.sigmoid(sal2), grey)

            loss += sal_loss1 + sal_loss2 + edge_loss

        loss = loss + pri_loss
        loss.backward()
        optimizer_Encoder.step()
        optimizer_Decoder.step()



        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, sal1 loss: {:0.4f}, edge loss: {:0.4f}, sal2 loss: {:0.4f}, prior loss: {:0.4f}'.
                  format(datetime.now(), epoch, args.epoch, i, total_step, loss.data, sal_loss1.data, edge_loss.data, sal_loss2.data, pri_loss.data))

            visualize_prediction(torch.sigmoid(sal1), './show/', "sal1")
            visualize_prediction(torch.sigmoid(edge_map), './show/', "edge")
            visualize_prediction(torch.sigmoid(sal2), './show/', "sal2")

    save_path = 'save_models/finetune/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if epoch % 1 == 0:
        torch.save(Encoder.state_dict(), save_path + 'scribble_Encoder' + '_%d'  % epoch  + '.pth')
        torch.save(Decoder.state_dict(), save_path + 'scribble_Decoder' + '_%d'  % epoch  + '.pth')

print("start training!!!")

for epoch in range(1, args.epoch+1):
    adjust_lr(optimizer_Encoder, epoch, args.decay_rate, args.decay_epoch)
    adjust_lr(optimizer_Decoder, epoch, args.decay_rate, args.decay_epoch)

    train(train_dataloader, Encoder, Decoder, optimizer_Encoder, optimizer_Decoder, epoch)


