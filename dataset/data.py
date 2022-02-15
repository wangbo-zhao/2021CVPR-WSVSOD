from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np






class VideoDataset(Dataset):
    def __init__(self, root_dir="/data1/zhaowangbo/weakly VSOD/", trainingset_list = ["DAVSOD", "DAVIS"],
                 video_time_clip=4, time_interval=1, trainging=True, transforms=None):
        super(VideoDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms

        self.time_clips = video_time_clip # must be <=4
        self.clips = []

        self.training = trainging

        for trainset in trainingset_list:

            video_root = os.path.join(root_dir, trainset)
            sequence_list = sorted(os.listdir(video_root))
            for sequence in sequence_list:
                sequence_info = self.get_frame_list(trainset, sequence)
                # print(len(sequence_info))
                self.clips += self.get_clips(sequence_info)

    def get_frame_list(self, trainset, sequence):
        image_path_root = os.path.join(self.root_dir, trainset, sequence, "Imgs")
        frame_list = sorted(os.listdir(image_path_root))
        sequence_info = []
        for i in range(len(frame_list)):
            frame_info = {"image_path": os.path.join(self.root_dir, trainset, sequence, "Imgs", frame_list[i]),
                          "gt_path": os.path.join(self.root_dir, trainset, sequence, "gt", frame_list[i]),
                          "mask_path": os.path.join(self.root_dir, trainset, sequence, "mask", frame_list[i]),
                          "Fixation_maps_smoothed": os.path.join(self.root_dir, trainset, sequence, "Fixation_maps_smoothed", frame_list[i]),
                          "Fixation_maps": os.path.join(self.root_dir, trainset, sequence, "Fixation_maps", frame_list[i]),
                          "GT_object_level": os.path.join(self.root_dir, trainset, sequence, "GT_object_level", frame_list[i]),
                          "flow": os.path.join(self.root_dir, trainset, sequence, "flow", frame_list[i]),
                          "grey": os.path.join(self.root_dir, trainset, sequence, "grey", frame_list[i])
                          }
            sequence_info.append(frame_info)

        return sequence_info


    def get_clips(self, sequence_info):
        clips = []
        for i in range(int(len(sequence_info) / self.time_clips)):
            clips.append(sequence_info[self.time_clips * i: self.time_clips * (i + 1)])

        finish = self.time_clips * (int(len(sequence_info) / self.time_clips))

        if finish < len(sequence_info):
            clips.append(sequence_info[len(sequence_info)-self.time_clips: len(sequence_info)])

        return clips

    def get_frame(self, frame_info):
        image_path = frame_info["image_path"]

        image = Image.open(image_path).convert("RGB")
        image_size = image.size[:2]


        if self.training:
            gt_path = frame_info["gt_path"]
            mask_path = frame_info["mask_path"]
            fixation_maps_path = frame_info["Fixation_maps"]
            fixation_maps_smoothed_path = frame_info["Fixation_maps_smoothed"]
            GT_object_level_path = frame_info["GT_object_level"]
            flow_path = frame_info["flow"]
            grey_path = frame_info["grey"]

            gt = Image.open(gt_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            fixation_maps = Image.open(fixation_maps_path).convert("L")
            fixation_maps_smoothed = Image.open(fixation_maps_smoothed_path).convert("L")
            GT_object_level = Image.open(GT_object_level_path).convert("L")
            flow = Image.open(flow_path).convert("RGB")
            grey = Image.open(grey_path).convert('L')


        else:
            gt = None
            mask = None
            fixation_maps = None
            fixation_maps_smoothed = None
            GT_object_level = None


        sample = {"image": image, "gt": gt, "mask": mask, "fixation_map": fixation_maps,
                  "fixation_map_smoothed": fixation_maps_smoothed,
                  "GT_object_level": GT_object_level,
                  "flow": flow, "grey":grey}


        sample["name"] = image_path
        sample["original_height"] = image_size[1]
        sample["original_width"] = image_size[0]

        return sample


    def __getitem__(self, idx):
        clip = self.clips[idx]

        clip_output = []
        #random revese when training
        if self.training and random.randint(0, 1):
            clip = clip[::-1]

        for i in range(len(clip)):
            item = self.get_frame(clip[i])
            clip_output.append(item)

        clip_output = self.transforms(clip_output)

        return clip_output

    def __len__(self):
        return len(self.clips)




















class ImageDataset(Dataset):
    def __init__(self, root_dir="/data1/zhaowangbo/weakly VSOD/", trainingset_list=["DUTS", "DAVSOD", "DAVIS"], image_transform=None):

        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = image_transform

        self.lists = []

        for trainset in trainingset_list:

            set_root = os.path.join(root_dir, trainset)
            self.lists += self.get_image_list(trainset, set_root)

    def get_image_list(self, trainset, set_root):

        image_list = sorted(os.listdir(os.path.join(set_root, "Imgs")))
        image_info = []

        for i in range(len(image_list)):
            frame_info = {"image_path": os.path.join(set_root, "Imgs", image_list[i]),
                          "gt_path": os.path.join(set_root, "gt", image_list[i]),
                          "mask_path": os.path.join(set_root, "mask", image_list[i]),
                          "flow": os.path.join(set_root, "flow", image_list[i]),
                          "grey": os.path.join(set_root, "grey", image_list[i])}

            image_info.append(frame_info)




        return image_info


    def __getitem__(self, idx):
        image_info = self.lists[idx]


        image_path = image_info["image_path"]
        gt_path = image_info["gt_path"]
        mask_path = image_info["mask_path"]
        flow_path = image_info["flow"]
        grey_path = image_info["grey"]


        image = Image.open(image_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image_size = image.size[:2]


        flow = Image.open(flow_path).convert("RGB")
        grey = Image.open(grey_path).convert("L")


        sample = {"image": image, "gt": gt, "mask": mask, "flow": flow, "grey": grey}

        sample = self.transforms(sample)


        sample["name"] = image_path
        sample["original_height"] = image_size[1]
        sample["original_width"] = image_size[0]

        return sample

    def __len__(self):
        return len(self.lists)









