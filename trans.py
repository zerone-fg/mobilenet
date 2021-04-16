#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 9:01
# @Author  : zerone
# @Site    : 
# @File    : trans.py
# @Software: PyCharm
import cv2 as cv
import os
file_path = "H:/color_new/c_12_1/"
for img in os.listdir(file_path):
    image = cv.imread(file_path + img)
    image_1 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    image_2 = cv.flip(image, 1)
    image_3 = cv.flip(image, 0)
    cv.imwrite(file_path + img.replace(".jpg", ".1.jpg"), image_1)
    cv.imwrite(file_path + img.replace(".jpg", ".2.jpg"), image_2)
    cv.imwrite(file_path + img.replace(".jpg", ".3.jpg"), image_3)
    print(img, " finished")
