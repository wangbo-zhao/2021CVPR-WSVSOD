from torch.utils.data import Dataset
import os
import random
from PIL import Image


class VideoDataset(Dataset):
    def __init__(self, root_dir="/data1/zhaowangbo/weakly-semi DAVSOD/testset/", video_time_clip=4, trainging=True, transforms=None):
        super(VideoDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms

        self.time_clips = video_time_clip # must be <=4
        self.clips = []

        self.training = trainging
        test_list = os.listdir(root_dir)
        for test_set in test_list:

            video_root = os.path.join(root_dir, test_set)
            sequence_list = sorted(os.listdir(video_root))
            for sequence in sequence_list:
                sequence_info = self.get_frame_list(test_set, sequence)
                # print(len(sequence_info))
                self.clips += self.get_clips(sequence_info)

    def get_frame_list(self, test_set, sequence):
        image_path_root = os.path.join(self.root_dir, test_set, sequence, "Imgs")
        frame_list = sorted(os.listdir(image_path_root))
        sequence_info = []
        for i in range(len(frame_list)):
            frame_info = {"image_path": os.path.join(self.root_dir, test_set, sequence, "Imgs", frame_list[i]),
                          "gt_path": os.path.join(self.root_dir, test_set, sequence, "gt", frame_list[i]),
                          "mask_path": os.path.join(self.root_dir, test_set, sequence, "mask", frame_list[i]),
                          "Fixation_maps_smoothed": os.path.join(self.root_dir, test_set, sequence, "Fixation_maps_smoothed", frame_list[i]),
                          "Fixation_maps": os.path.join(self.root_dir, test_set, sequence, "Fixation_maps", frame_list[i]),
                          "GT_object_level": os.path.join(self.root_dir, test_set, sequence, "GT_object_level", frame_list[i]),
                          "flow": os.path.join(self.root_dir, test_set, sequence, "flow", frame_list[i])
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

        flow_path = frame_info["flow"]
        flow = Image.open(flow_path).convert("RGB")

        gt = None
        mask = None
        fixation_maps = None
        fixation_maps_smoothed = None
        GT_object_level = None
        grey = None


        sample = {"image": image, "gt": gt, "mask": mask, "fixation_map": fixation_maps, "fixation_map_smoothed": fixation_maps_smoothed,
                  "GT_object_level": GT_object_level,
                  "flow": flow, "grey":grey}


        sample["name"] = image_path
        sample["original_height"] = image_size[1]
        sample["original_width"] = image_size[0]

        return sample


    def __getitem__(self, idx):
        clip = self.clips[idx]

        clip_output = []

        for i in range(len(clip)):
            item = self.get_frame(clip[i])
            clip_output.append(item)

        clip_output = self.transforms(clip_output)

        return clip_output


    def __len__(self):
        return len(self.clips)