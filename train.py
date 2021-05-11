#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 15:11
# @Author  : zerone
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mobilenetv3 import MobileNetV3_Large
from predataset import MyDataset
import matplotlib.pyplot as plt
from VGG_19 import VGG19
from predeal import get_set
import time
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomResizedCrop(224),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.5)),
    #transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser()  #实例化argumentparser
parser.add_argument("--epochs", default=100, help="the epochs you want to train")
parser.add_argument("--lr", default=0.0001, help="the rate to learn")
parser.add_argument("--batch_size", default=16, help="the batch you want to train")
parser.add_argument("--checkpoint", default="./外观，内饰，局部/mobilenet_appear.pth", help="the pretrained model")
args = parser.parse_args()


trainset = MyDataset(get_set(training="train"), transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

valset = MyDataset(get_set(training="valid"), transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = MobileNetV3_Large(num_classes=3)
model_dict = model.state_dict()

if args.checkpoint:
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
else:
    '''
    state_dict = torch.load('./vgg19-dcbb9e9d.pth')
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    #model.load_state_dict("./vgg19-dcbb9e9d.pth")
    '''
    model_dict = model.state_dict()
    state_dict = torch.load("./mbv3_large.old.pth.tar")
    for k, v in model_dict.items():
        temp = "module." + k
        if temp in state_dict["state_dict"] and "linear4" not in k:
            model_dict[k] = state_dict["state_dict"][temp]
    model.load_state_dict(model_dict)


if torch.cuda.device_count() > 1:
    print('We are using', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# 保存每个epoch后的Accuracy Loss Val_Accuracy
Accuracy = []
Loss = []
Val_Accuracy = []
BEST_VAL_ACC = 91

# 训练
since = time.time()
for epoch in range(args.epochs):
    train_loss = 0.
    train_accuracy = 0.
    run_accuracy = 0.
    run_loss = 0.
    total = 0.
    model.train()
    for i, data in enumerate(trainloader, 0):
        images, labels, path = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outs = model(images)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        run_loss += loss.item()
        _, prediction = torch.max(outs, 1)
        run_accuracy += (prediction == labels).sum().item()
        if i % 20 == 19:
            print('epoch {},iter {},train accuracy: {:.4f}%   loss:  {:.4f}'.format(epoch, i + 1, 100 * run_accuracy / (
                        labels.size(0) * 20), run_loss / 20))
            train_accuracy += run_accuracy
            train_loss += run_loss
            run_accuracy, run_loss = 0., 0.
    Loss.append(train_loss / total)
    Accuracy.append(100 * train_accuracy / total)
    # 可视化训练过程
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(0, epoch + 1, 1), Accuracy)
    ax1.set_title("Average trainset accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. train. accuracy")
    plt.savefig('Train_accuracy_vs_epochs.png')
    plt.clf()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(epoch + 1), Loss)
    ax2.set_title("Average trainset loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig('loss_vs_epochs.png')

    plt.clf()
    plt.close()
    # 验证
    acc = 0.
    model.eval()
    print('waitting for Val...')
    with torch.no_grad():
        accuracy = 0.
        total = 0
        for data in valloader:
            images, labels, rgb_path = data
            print(rgb_path)
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            _, prediction = torch.max(out, 1)
            total += labels.size(0)
            accuracy += (prediction == labels).sum().item()
            acc = 100. * accuracy / total
    print('epoch {}  The ValSet accuracy is {:.4f}% \n'.format(epoch, acc))
    Val_Accuracy.append(acc)
    if acc > BEST_VAL_ACC:
        print('Find Better Model and Saving it...')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/mobilenet_appear_16.55.pth')
        BEST_VAL_ACC = acc
        print('Saved!')

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch + 1), Val_Accuracy)
    ax3.set_title("Average Val accuracy vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Val accuracy")

    plt.savefig('val_accuracy_vs_epoch.png')
    plt.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Now the best val Acc is {:.4f}%'.format(BEST_VAL_ACC))
