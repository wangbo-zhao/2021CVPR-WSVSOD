import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF
import cv2
import argparse
from dataset.transforms import get_transforms
import os
from dataset.testdata import VideoDataset
from model.model import VideoEncoder, VideoDecoder
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser()


parser.add_argument('--size', default=256, type=int, help='image size')

args = parser.parse_args()


data_transforms = get_transforms(input_size=(args.size, args.size)
)

dataset = VideoDataset(root_dir='/data/nianliu/4Dsaliency/dataset/WV/testset/', trainging=False, transforms=data_transforms, video_time_clip=4)


Encoder = VideoEncoder(output_stride=16, pretrained=True)
Decoder = VideoDecoder()

Encoder.load_state_dict(torch.load("./save_models/finetune/scribble_Encoder_20.pth"))
Decoder.load_state_dict(torch.load("./save_models/finetune/scribble_Decoder_20.pth"))

Encoder = Encoder.cuda()
Decoder = Decoder.cuda()

frames = 0
total_time = 0
print("Begin inference on {} {}.")
for data in dataset:

    preds = []

    testset_name = data[0]["name"].split("/")[-4]
    sequence_name = data[0]["name"].split("/")[-3]
    save_dir = os.path.join("results", testset_name, sequence_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rgb_features = []
    flow_features = []

    torch.cuda.synchronize()
    start_time = time.time()

    for pack in data:
        images, flow = pack["image"], pack["flow"]
        images, flow = images.cuda().unsqueeze(0), flow.cuda().unsqueeze(0)


        rgb_feature, flow_feature = Encoder(images, flow)


        rgb_features.append(rgb_feature)
        flow_features.append(flow_feature)


    _, preds  = Decoder(rgb_features, flow_features)

    torch.cuda.synchronize()
    end_time = time.time()



    a = (end_time - start_time)

    total_time = total_time + a
    frames = frames +4


    if frames == 400:
        print("avg time", total_time / 400)


    for i in range(len(preds)):
        pred = preds[i]

        pred = 255 * torch.sigmoid(pred).data.cpu().squeeze().numpy()
        pred[pred >= 128] = 255
        pred[pred < 128] = 0

        image_name = data[i]["name"].split("/")[-1][:-3] + "png"
        image_forsave = cv2.resize(pred, (data[i]["original_width"], data[i]["original_height"]))
        cv2.imwrite(os.path.join(save_dir, image_name), image_forsave)
        print(os.path.join(save_dir, image_name))

