#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 15:29
# @Author  : zerone
# @Site    : 
# @File    : predeal.py
# @Software: PyCharm
import os
import sys
import random
from PIL import Image
from collections import defaultdict
classes = ["inside", "whole_appearance", "part_appearance"]   # label 0,1,2
direction = ["d_0", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7"]
color = ["c_0", "c_1", "c_3", "c_4", "c_5", "c_6", "c_7", "c_8", "c_9", "c_10", "c_11", "c_12"]
dict_count = {}
def generate_txt_1():
    '''

    :return: 按照8：1：1生成训练集，验证集和测试集
    '''
    train_set = []
    valid_set = []
    test_set = []
    for i in range(15):
        dict_count[i] = 0
    path_list = os.listdir("H:/chexi_download/")
    #path_list.sort(key=lambda x: int(x.split('.')[0]))
    for path in path_list:
        path = path.split(".")[-1]
        '''
        if path[-1] == "5":
            label = 1
            continue
        elif path[-1] == "6":
            label = 2
            continue
        elif path[-1] == "7":
            label = 0
            continue
        '''
        if path.startswith("d"):
            label = 0
            label = int(path.split("_")[-1])
        else:
            continue
        _path = os.path.join("H:/chexi_download", path)
        for img in os.listdir(_path):
            dict_count[label] += 1
            img_path = os.path.join(_path, img)
            temp = random.random()
            if temp <= 0.8:
                train_set.append((img_path, label))
            elif temp > 0.8 and temp < 0.9:
                valid_set.append((img_path, label))
            else:
                test_set.append((img_path, label))
    with open("./train_1.txt", "w") as f1:
        for item in train_set:
            f1.write(item[0] + " " + str(item[1]) + "\n")
    f1.close()
    with open("./valid_1.txt", "w") as f2:
        for item in valid_set:
            f2.write(item[0] + " " + str(item[1]) + "\n")
    f2.close()
    with open("./test_1.txt", "w") as f3:
        for item in test_set:
            f3.write(item[0] + " " + str(item[1]) + "\n")
    f3.close()
    print(dict_count[0], dict_count[1], dict_count[2])
def get_set(training):
    '''

    :param training: 指示符，"train","valid","test"
    :return: 返回对应的dataset
    '''
    set_list = []
    if training == "train":
        with open("./外观，内饰，局部/train_1.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                try:
                    img = Image.open(line.split(" ")[0])
                    set_list.append({"img": line.split(" ")[0], "label":line.split(" ")[1]})
                except:
                    continue
        f.close()
    elif training == "valid":
        with open("./外观，内饰，局部/valid_1.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                try:
                    img = Image.open(line.split(" ")[0])
                    set_list.append({"img": line.split(" ")[0], "label": line.split(" ")[1]})
                except:
                    continue
        f.close()
    else:
        with open("./外观，内饰，局部/test_1.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                try:
                    img = Image.open(line.split(" ")[0])
                    set_list.append({"img": line.split(" ")[0], "label": line.split(" ")[1]})
                except:
                    continue
        f.close()
    return set_list
if __name__ =="__main__":
    generate_txt_1()


