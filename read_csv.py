#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/31 15:33
# @Author  : zerone
# @Site    : 
# @File    : read_csv.py
# @Software: PyCharm
import csv
import pandas as pd
import numpy as np
def check_csv():
    '''
    with open("H:/外观、前排、后排.csv", "r", encoding="gbk") as fi:
        csv_reader = csv.reader(fi)
        headers = next(csv_reader)
        hid = {n: i for i, n in enumerate(headers)}
        print(hid["color_id"])
    '''
    csv = pd.read_csv("H:/外观、前排、后排.csv", encoding="gbk")
    a = csv["color_id"]
    b = csv["photo_name"]
    dict = {_b: _a for _b, _a in zip(b, a)}
    with open("./check.txt", "a") as f:
        for k, v in dict.items():
            f.write(k + " " + str(v) + "\n")
    f.close()
def read_txt():
    dict_count={}
    dict_count['0'] = 0
    dict_count['1'] = 0
    dict_count['2'] = 0
    with open("./train.txt", "r") as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.strip()
            label = line[-1]
            dict_count[label] += 1
    f1.close()
    print("train_set:", dict_count['0'],dict_count['1'],dict_count['2'])
    dict_count['0'] = 0
    dict_count['1'] = 0
    dict_count['2'] = 0
    with open("./valid.txt", "r") as f2:
        lines = f2.readlines()
        for line in lines:
            line = line.strip()
            label = line[-1]
            dict_count[label] += 1
    f2.close()
    print("valid_set:", dict_count['0'], dict_count['1'], dict_count['2'])
    dict_count['0'] = 0
    dict_count['1'] = 0
    dict_count['2'] = 0
    with open("./test.txt", "r") as f3:
        lines = f3.readlines()
        for line in lines:
            line = line.strip()
            label = line[-1]
            dict_count[label] += 1
    print("test_set:", dict_count['0'], dict_count['1'], dict_count['2'])
    f3.close()
if __name__=="__main__":
    check_csv()