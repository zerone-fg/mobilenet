#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 15:03
# @Author  : zerone
# @Site    : 
# @File    : predataset.py
# @Software: PyCharm
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
class MyDataset(Dataset):
    def __init__(self, set_list, transform):
        self.transform = transform
        self.imgs = [item["img"] for item in set_list]
        self.label = [item["label"] for item in set_list]
    def __getitem__(self, index):
        img_rgb_path = self.imgs[index]
        img_rgb = Image.open(img_rgb_path)
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
        return img_rgb, int(self.label[index]), img_rgb_path
    def __len__(self):
        return len(self.imgs)



