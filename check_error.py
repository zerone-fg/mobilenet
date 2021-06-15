#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/19 13:44
# @Author  : zerone
# @Site    : 
# @File    : check_error.py
# @Software: PyCharm
import os
import logging
import urllib
import traceback
import time
import csv
import pandas as pd
import xlrd
import urllib.request
def download(img_url, save_path):
    '''download'''
    strlog = img_url + "\t" + save_path
    if os.path.isfile(save_path) and os.path.getsize(save_path) > 10240:
        logging.debug(' [exist] ' + strlog)
        return -1
    try:
        fi = urllib.request.urlopen(img_url)
        img_file = fi.read()
        # img_file = urllib.request.urlretrieve()
    except:
        traceback.print_exc()
        logging.debug(' [urlopen error] ' + strlog)
        return -2
    else:
        with open(save_path, "wb") as fo:
            fo.write(img_file)
            logging.info(' [write success] ' + strlog)
    return 0
def main(input):
    count = 0
    logging.basicConfig(
        filename=time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_download_pic.log",
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    prefix = "H:/error_data/"
    df = pd.read_excel(input)
    items = df[["id", "image_url", "code"]]
    for i in range(len(df)):
        id = str(df["id"][i])
        code = df["code"][i]
        url_path = df["image_url"][i]
        if code != 0:
            save_path = os.path.join(prefix, "{}_{}.jpg".format(id, code))
            ret = download(url_path, save_path)
            if ret == 0:
                print(id, " finished downloading")
            else:
                print("{} is none".format(id))
        count+=1
    print(count)
if __name__=="__main__":
    main("D:/20210518.xlsx")