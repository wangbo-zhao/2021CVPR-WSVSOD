import random

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils import data
import numpy as np



def get_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    return transforms.Compose([Resize(input_size),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])

def get_train_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([Resize(input_size),
                               RandomFlip(),
                               Random_crop(15),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])

def get_imagetrain_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([ImageResize(input_size),
                               ImageRandomFlip(),
                               ImageRandom_crop(15),
                               ImageToTensor(),
                               ImageNormalize(mean=mean, std=std)])











class RandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    """
    def __call__(self, samples):
        rand_flip_index = random.randint(0, 1)

        if rand_flip_index == 1:

            for i in range(len(samples)):
                sample = samples[i]

                image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey = sample['image'], \
                                                                                        sample['gt'], \
                                                                                        sample['mask'], \
                                                                                        sample['fixation_map'], \
                                                                                        sample['fixation_map_smoothed'], \
                                                                                        sample["GT_object_level"], \
                                                                                        sample["flow"], \
                                                                                        sample["grey"]



                image = F.hflip(image)
                flow  = F.hflip(flow)
                gt = F.hflip(gt)
                mask = F.hflip(mask)
                fixation_map = F.hflip(fixation_map)
                fixation_map_smoothed = F.hflip(fixation_map_smoothed)
                GT_object_level = F.hflip(GT_object_level)
                grey = F.hflip(grey)

                sample['image'], sample['gt'], sample['mask'], sample['fixation_map'], sample['fixation_map_smoothed'], sample["GT_object_level"], \
                   sample["flow"], sample["grey"] = image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey

                samples[i] = sample


        else:

            pass

        return samples

class Resize(object):
    """ Resize PIL image use both for training and inference"""
    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey = \
                sample['image'], sample['gt'], sample['mask'], sample['fixation_map'], sample['fixation_map_smoothed'], \
                sample["GT_object_level"], sample["flow"], sample["grey"]


            image = F.resize(image, self.size, Image.BILINEAR)
            flow = F.resize(flow, self.size, Image.BILINEAR)

            if gt is not None:
                gt = F.resize(gt, self.size, Image.NEAREST)
                mask = F.resize(mask, self.size, Image.NEAREST)
                fixation_map = F.resize(fixation_map, self.size, Image.NEAREST)
                fixation_map_smoothed = F.resize(fixation_map_smoothed, self.size, Image.NEAREST)
                GT_object_level = F.resize(GT_object_level, self.size, Image.NEAREST)
                grey = F.resize(grey, self.size, Image.BILINEAR)

            sample['image'], sample['gt'], sample['mask'], \
            sample['fixation_map'], sample['fixation_map_smoothed'], sample["GT_object_level"], sample["flow"], sample["grey"] = \
                image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey

            samples[i] = sample

        return samples

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):

        for i in range(len(samples)):
            sample = samples[i]
            image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey = \
                sample['image'], sample['gt'], sample['mask'], sample['fixation_map'], sample['fixation_map_smoothed'], \
                sample["GT_object_level"], sample["flow"], sample["grey"]


            image = F.to_tensor(image)
            flow = F.to_tensor(flow)

            if gt is not None:
                gt = F.to_tensor(gt)
                mask = F.to_tensor(mask)
                fixation_map = F.to_tensor(fixation_map)
                fixation_map_smoothed = F.to_tensor(fixation_map_smoothed)
                GT_object_level = F.to_tensor(GT_object_level)
                grey = F.to_tensor(grey)

            sample['image'], sample['gt'], sample['mask'], \
            sample['fixation_map'], sample['fixation_map_smoothed'], sample["GT_object_level"], sample["flow"], sample["grey"] = \
                image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey

            samples[i] = sample

        return samples

class Normalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) ? Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """
    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, samples):

        for i in range(len(samples)):
            sample = samples[i]


            image = sample['image']
            flow = sample["flow"]

            image = F.normalize(image, self.mean, self.std)
            flow = F.normalize(flow, self.mean, self.std)

            sample["image"] = image
            sample["flow"] = flow

            samples[i] = sample

        return samples


class Random_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, samples):

        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)

        width, height = samples[0]["image"].size
        assert  samples[0]["image"].size == samples[0]["gt"].size
        region = [x, y, width - x, height - y]

        for i in range(len(samples)):
            sample = samples[i]

            image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey = sample['image'], \
                                                                                          sample['gt'], \
                                                                                          sample['mask'], \
                                                                                          sample['fixation_map'], \
                                                                                          sample['fixation_map_smoothed'], \
                                                                                          sample["GT_object_level"], \
                                                                                          sample["flow"], \
                                                                                          sample["grey"]

            image = image.crop(region)
            flow = flow.crop(region)
            gt = gt.crop(region)
            mask = mask.crop(region)
            fixation_map = fixation_map.crop(region)
            fixation_map_smoothed = fixation_map_smoothed.crop(region)
            GT_object_level = GT_object_level.crop(region)
            grey = grey.crop(region)


            image = image.resize((width, height), Image.BILINEAR)
            flow = flow.resize((width, height), Image.BILINEAR)
            gt = gt.resize((width, height), Image.NEAREST)
            mask = mask.resize((width, height), Image.NEAREST)
            fixation_map = fixation_map.resize((width, height), Image.NEAREST)
            fixation_map_smoothed = fixation_map_smoothed.resize((width, height), Image.NEAREST)
            GT_object_level = GT_object_level.resize((width, height), Image.NEAREST)
            grey = grey.resize((width, height), Image.BILINEAR)

            sample['image'], sample['gt'], sample['mask'], \
            sample['fixation_map'], sample['fixation_map_smoothed'], sample["GT_object_level"], sample["flow"], sample["grey"] = \
                image, gt, mask, fixation_map, fixation_map_smoothed, GT_object_level, flow, grey

            samples[i] = sample

        return samples






































class ImageResize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, gt, mask, flow, grey = sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"]

        image = F.resize(image, self.size, Image.BILINEAR)
        flow = F.resize(flow, self.size, Image.BILINEAR)
        gt = F.resize(gt, self.size, Image.NEAREST)
        mask = F.resize(mask, self.size, Image.NEAREST)
        grey = F.resize(grey, self.size, Image.BILINEAR)

        sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"] = image, gt, mask, flow, grey

        return sample



class ImageRandom_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):

        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)

        width, height = sample["image"].size
        assert  sample["image"].size == sample["gt"].size
        region = [x, y, width - x, height - y]

        image, gt, mask, flow, grey = sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"]

        image = image.crop(region)
        flow = flow.crop(region)
        gt = gt.crop(region)
        mask = mask.crop(region)
        grey = grey.crop(region)


        image = image.resize((width, height), Image.BILINEAR)
        flow = flow.resize((width, height), Image.BILINEAR)
        gt = gt.resize((width, height), Image.NEAREST)
        mask = mask.resize((width, height), Image.NEAREST)
        grey = grey.resize((width, height), Image.BILINEAR)

        sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"] = image, gt, mask, flow, grey

        return sample


class ImageRandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    """
    def __call__(self, sample):
        rand_flip_index = random.randint(0, 1)

        if rand_flip_index == 1:

            image, gt, mask, flow, grey = sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"]

            image = F.hflip(image)
            flow  = F.hflip(flow)
            gt = F.hflip(gt)
            mask = F.hflip(mask)
            grey = F.hflip(grey)

            sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"] = image, gt, mask, flow, grey

        return sample


class ImageToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt, mask, flow, grey = sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"]

        image = F.to_tensor(image)
        flow = F.to_tensor(flow)
        gt = F.to_tensor(gt)
        mask = F.to_tensor(mask)
        grey = F.to_tensor(grey)

        sample['image'], sample['gt'], sample['mask'], sample["flow"], sample["grey"] = image, gt, mask, flow, grey

        return sample



class ImageNormalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) ? Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """
    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, flow = sample['image'], sample["flow"]

        image = F.normalize(image, self.mean, self.std)
        flow = F.normalize(flow, self.mean, self.std)

        sample["image"] = image
        sample["flow"] = flow

        return sample






