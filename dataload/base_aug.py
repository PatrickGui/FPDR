
import cv2
import os
import random
import shutil
import numpy as np
from PIL import Image

from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from background_mixing import *
from random_crop import RandomCropping

normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

def preprocess(normalize, image_size):
    return transforms.Compose([
        # transforms.CenterCrop((256,256)),

        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),
        normalize
    ])

def preprocess_baseline(normalize, image_size):
    return transforms.Compose([
        # RandomChangeBack(p=0.3, back_list_path='/raid_new/GPH/DataSet/back_list_9'),######Background Mixing
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        # transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),    #####Random Cropping

        transforms.Resize((image_size, image_size)),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_baseline_RCR(normalize, image_size):
    return transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        # RandomCropping(size=(448,448),scale=(0.5,1.0)),

        transforms.Resize((image_size, image_size)),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_baseline_AddBack(normalize, image_size):
    return transforms.Compose([
        RandomChangeBack(p=0.3, back_list_path='./dataload/background_img/back_list_9'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        # transforms.RandomResizedCrop(image_size, scale=(0.1, 1.0)),
        # transforms.RandomCrop((image_size, image_size)),
        transforms.Resize((image_size, image_size)),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_baseline_RCR_AddBack(normalize, image_size):
    return transforms.Compose([
        RandomChangeBack(p=0.3, back_list_path='./dataload/background_img/back_list_9'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        # RandomCropping(size=(448,448),scale=(0.5,1.0)),

        transforms.Resize((image_size, image_size)),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])