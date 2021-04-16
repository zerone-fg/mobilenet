#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/15 23:39
# @Author  : zerone
# @Site    :
# @File    : aa.py
# @Software: PyCharm
import os
import time
import argparse
# import logging
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

# __dir__ = os.path.dirname(os.path.abspath(__file__))


class Detector:
    def __init__(self, gpu_id=0, threshold=0.5):
        self.device = torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold

    def detect_car(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        # image = torch.from_numpy(img)
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        print(image.size())
        with torch.no_grad():
            predictions = self.model(image)[0]
            # print(predictions)

        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        if len(boxes) < 1:
            return None, 0

        indies = np.where(np.isin(labels, [3,6,8]))
        # indies = np.where(labels == 3)
        # if len(indies[0]) < 1:
        #     print("no car")
        #     indies = np.where(labels == 6)
        # if len(indies[0]) < 1:
        #     print("no bus")
        #     indies = np.where(labels == 8)
        # if len(indies[0]) < 1:
        #     print("no truck")
        #     return None, 0
        if len(indies[0]) < 1:
            print("no car/bus/truck")
            return None, 0
        boxes = boxes[indies]
        scores = scores[indies]
        labels = labels[indies]
        # print("boxes    :{}".format(boxes))
        # print("labels   :{}".format(labels))
        # print("scores   :{}".format(scores))

        max_box = np.array(boxes[0]).astype(np.int32).tolist()
        max_area = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        max_score = scores[0]
        if len(boxes) > 1:
            for i in range(1, len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2 + 0.5), int(y2 + 0.5)
                area = (x2 - x1) * (y2 - y1)
                if scores[i] > max_score + 0.5 or (area > max_area and scores[i] + 0.3 > max_score):
                    max_area = area
                    max_box = [x1, y1, x2, y2]
                    max_score = scores[i]
        return max_box, float(max_score)

    def detect_car_multi(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        # image = torch.from_numpy(img)
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        print(image.size())
        with torch.no_grad():
            predictions = self.model(image)[0]
            # print(predictions)

        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        if len(boxes) < 1:
            return None, 0

        indies = np.where(np.isin(labels, [3,6,8]))
        if len(indies[0]) < 1:
            print("no car/bus/truck")
            return None, 0
        boxes = boxes[indies]
        scores = scores[indies]
        labels = labels[indies]
        # print("boxes    :{}".format(boxes))
        # print("labels   :{}".format(labels))
        # print("scores   :{}".format(scores))

        box_list = []
        score_list = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2 + 0.5), int(y2 + 0.5)
            if x2 - x1 > 50 and y2 - y1 > 50 and scores[i] >= self.threshold:
                box_list.append((x1, y1, x2, y2))
                score_list.append(float(scores[i]))
        return box_list, score_list

def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    #img_name = os.path.basename(args.input)
    det = Detector()
    for img in os.listdir(args.input):
        image = cv2.imread(args.input + img)
        print("只识别一个车")
        try:
            box, score = det.detect_car(image)
        except:
            continue
        print(box, score)
        img_show = image.copy()
        cropped = img_show[box[1]:box[3], box[0]:box[2]]
        #cv2.rectangle(img_show, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=4)
        cv2.imwrite(os.path.join(args.out_dir, img), cropped)


if __name__ == '__main__':
    #main(get_args())
    for a in range(8):
        parser = argparse.ArgumentParser()  # 实例化argumentparser
        parser.add_argument("--input", default="H:/chexi_download/d_{}/".format(a))
        parser.add_argument("--out_dir", default="H:/d_{}_1/".format(a))
        args = parser.parse_args()
        main(args)
        print(a, "finished")