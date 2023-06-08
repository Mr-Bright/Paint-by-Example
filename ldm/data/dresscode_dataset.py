from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A



def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

# 两个的初始化不一样
def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
        transform_list += [torchvision.transforms.Resize((224,224))]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

class DressCodeTestDataset(data.Dataset):
    
    def __init__(self):
        super(DressCodeTestDataset, self).__init__()
        
        self.original_img_dir = '/home/kwang/DD_project/SCHP/input/'
        self.body_parser_dir = '/home/kwang/DD_project/SCHP/label_output/mask/'
        self.densepose_dir = '/home/kwang/DD_project/DressCode/upper_body/dense/'
        self.cut_warped_cloth_dir = '/home/kwang/DD_project/C-VTON/arugment_output/cut_clothes/'

        self.data_txt_path = '/home/kwang/DD_project/DressCode/upper_body/test_pairs_paired.txt'

        self.data = []
        self.test_cloth = []
        with open(self.data_txt_path, 'rt') as f:
            for line in f:
                self.data.append(line.split('\t')[0])
                self.test_cloth.append(line.split('\t')[1].strip('\n'))
                # self.test_cloth.append(line.split('\t')[0])
        
    def name(self):
        return "DressCodeArugmentDataset"
    
    def __getitem__(self, index):
        img_name = self.data[index]
        cloth_name = self.test_cloth[index]

        img_p = Image.open(self.original_img_dir + img_name).convert("RGB")

        image_tensor = get_tensor(normalize=True, toTensor=True)(img_p)

        # 处理去除原有衣服的图片
        body_parser = cv2.imread(self.body_parser_dir + img_name[:-4] + '.png',0)
        densepose = cv2.imread(self.densepose_dir + img_name[:-6] + '_5.png', 0)
        # todo 可以再调整 1,2是躯干
        person_cloth_body = ((body_parser==4) | (np.isin(densepose, [15,17,16,18,19,21,20,22]))).astype(np.uint8)

        reference_area = (body_parser==4) | (np.isin(densepose, [1,2,15,17,16,18,19,21,20,22]))

        # 用高斯噪声填充原有衣服和人体部分
        img_array = np.array(img_p)/127.5 - 1
        person_cloth_body = person_cloth_body[:,:,np.newaxis]
        person_cloth_body = np.concatenate([person_cloth_body, person_cloth_body, person_cloth_body], axis=2)
        random_noise = np.random.normal(0, 1, img_array.shape)
        inpaint_img = img_array * (1 - person_cloth_body) + random_noise * person_cloth_body
        inpaint_img = get_tensor()(Image.fromarray(((inpaint_img + 1) * 127.5).astype(np.uint8)))

        # 指示合成区域
        reference_area = 1-get_tensor(normalize=False, toTensor=True)(Image.fromarray(reference_area.astype(np.uint8)*255))

        # 处理clip的指定衣服图片

        clip_img = Image.open(self.cut_warped_cloth_dir + cloth_name).convert("RGB")
        clip_img = get_tensor_clip()(clip_img)

        item = {
            "GT":image_tensor,
            "inpaint_image":inpaint_img,
            "inpaint_mask":reference_area,
            "ref_imgs":clip_img,
            "filename":img_name[:-4]+"_"+cloth_name[:-4]
            }
        return item
    
    def __len__(self):
        return len(self.data)