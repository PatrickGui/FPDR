import json
import os

import torch
from torch.utils.data import *
from PIL import Image
from torchvision.transforms import transforms
IMAGE_SIZE=224

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataSet(Dataset):
    def __init__(self, root, loader=default_loader, extensions=None, transform=None, target_transform=None, root_pre=None, augment=False):
        samples = root
        self.samples = samples  #2250
        self.targets = [int(s[1]) for s in samples]
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions
        self.root_pre = root_pre
        self.augment = augment

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(self.root_pre+path)
        if self.transform is not None:
            sample = self.transform(sample)
        if not torch.is_tensor(sample):
            t = transforms.Compose([
                transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            sample = t(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
class MyDataSet_Sp(Dataset):
    def __init__(self, root, loader=default_loader, extensions=None, transform=None, target_transform=None, root_pre=None, augment=False):

        samples = root
        self.samples = samples  #2250
        self.targets = [s[1] for s in samples]
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions
        self.root_pre = root_pre
        self.augment = augment

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(self.root_pre+path)
        if self.transform is not None:
            sample = self.transform(sample)
        if not torch.is_tensor(sample):
            t = transforms.Compose([
                transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            sample = t(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
