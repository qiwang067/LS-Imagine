import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import Normalize

MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)
MC_NORMALIZER = Normalize(mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)

def MCRandomHorizontalFlip(image, label, p=0.5):
    if np.random.rand() < p:
        image = image[:, ::-1, :]
        label = label[:, ::-1, :]
    return image, label

def apply_affine(image, angle, translate, scale, border_mode):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += translate[0] * w
    M[1, 2] += translate[1] * h

    if len(image.shape) == 2 or image.shape[2] == 1: 
        transformed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
        transformed_image = np.expand_dims(transformed_image, axis=2) 
    else:
        transformed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)

    return transformed_image

def MCRandomAffine(image, label, p=0.7, angle=(-5, 5), translate=(0.05, 0.05), scale=(1.05, 1.15), border_mode=cv2.BORDER_REFLECT):
    if np.random.rand() < p:
        angle = random.uniform(angle[0], angle[1])
        translate = (random.uniform(-translate[0], translate[0]),
                     random.uniform(-translate[1], translate[1]))
        scale = random.uniform(scale[0], scale[1])

        image = apply_affine(image, angle, translate, scale, border_mode)
        label = apply_affine(label, angle, translate, scale, border_mode)

    return image, label

def resized_if_need(image, target_size=(256, 160)):
    if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
    return image
    
def MCResize(image, label, target_size=(256, 160)):
    image = resized_if_need(image, target_size)
    label = resized_if_need(label, target_size)
    
    return image, label

def MyToTensor(image, label):
    return torch.tensor(image.copy()).permute(2, 0, 1), torch.tensor(label.copy()).permute(2, 0, 1)
    
def MNormalize(image, label):
    return MC_NORMALIZER(image / 255.0), label

class RandomGenerator(object):
    def __init__(self, output_height=160, output_width=256):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, sample):
        image, label, prompt = sample['image'], sample['label'], sample['prompt']
        # image.shape = (160, 256, 3)
        # label.shape = (160, 256, 1)

        image, label = MCRandomHorizontalFlip(image, label)
        image, label = MCRandomAffine(image, label)
        image, label = MCResize(image, label, target_size=(self.output_width, self.output_height))
        image, label = MyToTensor(image, label)

        org_img = image

        image, label = MNormalize(image, label)

        prompt = torch.tensor(prompt)

        sample = {'image': image, 'label': label, 'prompt': prompt, 'org_img': org_img}
        return sample
    
class Minecraft_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            npz_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, npz_name+'.npz')
            data = np.load(data_path)

            image, label, prompt = data['image'], data['label'], data['prompt']

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label, prompt = data['image'][:], data['label'][:], data['prompt'][:]

        sample = {'image': image, 'label': label, 'prompt': prompt}
        
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

