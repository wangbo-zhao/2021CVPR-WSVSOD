import os
import torch
import torch.nn as nn
from torch.utils import data
from datetime import datetime
import argparse
from dataset.data import ImageDataset
from dataset.transforms import get_imagetrain_transforms
from torch.utils.data import DataLoader
from model.model import VideoEncoder, VideoDecoder
import torch.optim as optim
from utils import adjust_lr, label_edge_prediction, visualize_prediction
from datetime import datetime
import smoothness


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
args = parser.parse_args()

transforms = get_imagetrain_transforms(input_size=(args.size, args.size))
dataset = ImageDataset(root_dir="the dir of your dataset", trainingset_list=["DUTS"], image_transform=transforms)
train_dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize, num_workers=6, shuffle=True, drop_last=True)

Encoder = VideoEncoder(output_stride=16, pretrained=True)
Decoder = VideoDecoder()

Encoder = Encoder.cuda()
Decoder = Decoder.cuda()
print("network ready!!!")

optimizer_Encoder = optim.Adam(Encoder.parameters(), lr=args.lr)
optimizer_Decoder = optim.Adam(Decoder.parameters(), lr=args.lr)

CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)

def train(train_loader, Encoder, Decoder, optimizer_Encoder, optimizer_Decoder, epoch):
    Encoder.train()
    Decoder.train()

    total_step = len(train_dataloader)
    for i, pack in enumerate(train_dataloader):
        optimizer_Encoder.zero_grad()
        optimizer_Decoder.zero_grad()

        images, gts, masks, flow, grey = pack["image"], pack["gt"], pack["mask"], pack["flow"], pack["grey"]
        images, gts, masks, flow, grey = images.cuda(), gts.cuda(), masks.cuda(), flow.cuda(), grey.cuda()


        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        rgb_features, flow_features = Encoder(images, flow)
        edge_map = rgb_features[-1]
        sal1, sal2 = Decoder(rgb_features, flow_features)

        sal1_prob = torch.sigmoid(sal1)
        sal1_prob = sal1_prob * masks
        sal2_prob = torch.sigmoid(sal2)
        sal2_prob = sal2_prob * masks

        edges_gt = label_edge_prediction(torch.sigmoid(sal2).detach())
        edge_loss = CE(torch.sigmoid(edge_map), edges_gt)

        sal_loss1 = ratio * CE(sal1_prob, gts*masks) + 0.3 * smooth_loss(torch.sigmoid(sal1), grey)
        sal_loss2 = ratio * CE(sal2_prob, gts*masks) + 0.3 * smooth_loss(torch.sigmoid(sal2), grey)


        loss = sal_loss1 + sal_loss2 + edge_loss
        loss.backward()
        optimizer_Encoder.step()
        optimizer_Decoder.step()



        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, sal1 loss: {:0.4f}, edge loss: {:0.4f}, sal2 loss: {:0.4f}'.
                  format(datetime.now(), epoch, args.epoch, i, total_step, loss.data, sal_loss1.data, edge_loss.data, sal_loss2.data))

            visualize_prediction(torch.sigmoid(sal1), './show/', "sal1")
            visualize_prediction(torch.sigmoid(edge_map), './show/', "edge")
            visualize_prediction(torch.sigmoid(sal2), './show/', "sal2")

    save_path = 'save_models/pretrained/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if epoch % 10 == 0:
        torch.save(Encoder.state_dict(), save_path + 'scribble_Encoder' + '_%d'  % epoch  + '.pth')
        torch.save(Decoder.state_dict(), save_path + 'scribble_Decoder' + '_%d'  % epoch  + '.pth')

print("start training!!!")

for epoch in range(1, args.epoch+1):
    adjust_lr(optimizer_Encoder, epoch, args.decay_rate, args.decay_epoch)
    adjust_lr(optimizer_Decoder, epoch, args.decay_rate, args.decay_epoch)

    train(train_dataloader, Encoder, Decoder, optimizer_Encoder, optimizer_Decoder, epoch)



