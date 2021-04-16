#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 15:13
# @Author  : zerone
# @Site    : 
# @File    : visualization.py
# @Software: PyCharm
import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

trainset = torchvision.datasets.ImageFolder('../data/val/', transform=transform)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4, shuffle=True)
import matplotlib.pyplot as plt
#%matplotlib inline

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(labels)