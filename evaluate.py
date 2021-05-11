#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/30 10:02
# @Author  : zerone
# @Site    : 
# @File    : evaluate.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenetv3 import MobileNetV3_Large
import torchvision.transforms as transforms
from predeal import get_set
from VGG_19 import VGG19
import matplotlib.pyplot as plt
import numpy as np
from predataset import MyDataset
from PIL import Image
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
model = MobileNetV3_Large(num_classes=3)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)
model.load_state_dict(torch.load('./checkpoint/mobilenet_appear_16.55.pth'))
# 测试
id_list = []
pred_list = []
testset = MyDataset(get_set(training="test"), transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

model.eval()
acc = 0
avg_acc = 0
with torch.no_grad():
    #f = open("./diff.txt", "a")
    for i, data in enumerate(testloader):
        images, labels, rgb_path = data
        _id = labels
        img = images.to(device)
        out = model(img)
        _, prediction = torch.max(out, 1)
        if _id.item() == prediction.item():
            acc += 1
        else:
            print(rgb_path)
            #f.write(rgb_path[0] + "\n")
        id_list.append(_id)
        pred_list.append(prediction)
        print(_id.item(), prediction.item(), i)
    #f.close()
avg_acc = acc / len(testloader)

print("test finished, the accuracy is {}".format(avg_acc))
