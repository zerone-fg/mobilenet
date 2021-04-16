
import os
import argparse
import csv
import urllib
import urllib.request
import logging
import time
import traceback
import random   #0  1      2     3    4     5     6     7    8      9     10    11     12
ColorList = ["银", "黑", "绿", "橙", "白", "灰", "红", "蓝", "紫", "黄", "金", "棕", "咖啡"]
DirectList = ['侧前45度车头向右水平', '侧前45度车头向左水平', '侧后45度车头向右水平', '侧后45度车头向左水平', '正侧车头向右水平', '正侧车头向左水平', '正前水平',
                         '正后水平']
def check_dire(pr_name):
    '''
    确定汽车朝向
    :param pr_name:
    :return:
    '''
    for i, p in enumerate(DirectList):
        if pr_name == p:
            direction = i
            return direction
    return -1
def check_color(p_name):
    '''
    在字符串中查找颜色
    :param p_name:
    :return:
    '''
    p_list = p_name.split(" ")
    for s in p_list:
        if s.startswith("外"):
            for index, c in enumerate(ColorList):
                if s.find(c) != -1:
                    return index
    return -1
def str2int(s):
    '''str to int'''
    try:
        return int(s)
    except ValueError:
        traceback.print_exc()
        return s


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='H:/外观、前排、后排.csv')
    parser.add_argument('--out_dir', default='H:/chexi_download/')
    return parser.parse_args()


def main(args):
    logging.basicConfig(
        filename=time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_download_pic.log",
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    url_prefix = 'https://img1.bitautoimg.com/autoalbum/'

    with open(args.input, 'r', encoding="gbk") as fi:
        csv_reader = csv.reader(fi)
        headers = next(csv_reader)
        # headers = fi.readline().strip().split(',')
        print(headers)
        hid = {n:i for i, n in enumerate(headers)}
        print(hid)
        dict_count = {}
        dict_count['5'] = 0
        dict_count['6'] = 0
        dict_count['7'] = 0
        for row in csv_reader:
            if len(row) != len(headers):
                print("len(row) != len(headers)")
                continue
            url_path = url_prefix + row[hid['path']]
            main_brand_id = str(int(row[hid['main_brand_id']]))
            main_brand_name = row[hid['main_brand_name']]
            brand_id = str(int(row[hid['brand_id']]))
            brand_name = row[hid['brand_name']]
            # series_id = str(int(row[hid['series_id']]))
            series_id = row[hid['series_id']]
            series_name = row[hid['series_name']]
            # sell_year = str(int(row[hid['sell_year']]))
            sell_year = row[hid['sell_year']]
            spec_id = str(int(row[hid['spec_id']]))
            spec_name = row[hid['spec_name']]
            photo_id = str(int(row[hid['photo_id']]))
            photo_name = row[hid['photo_name']]
            property_id = row[hid['property_group_id']]
            property_name = row[hid['property_name']]
            sid_int = str2int(series_id)
            if not isinstance(sid_int, int):
                print("series_id is not a integer !!!!", row, series_id)
                continue
            if property_id == "6":
                direction = check_dire(property_name)
                if direction != -1:
                    #color = check_color(photo_name)
                    #out_dir_1 = os.path.join(args.out_dir, "{}.{}.{}.{}.d_{}".format(main_brand_id, brand_id, series_id, property_id, direction))
                    out_dir_1 = os.path.join(args.out_dir, "d_{}".format(direction))
                    if not os.path.isdir(out_dir_1):
                        os.makedirs(out_dir_1)
                    save_path_1 = os.path.join(out_dir_1, "{}.{}.{}.jpg".format(sell_year, spec_id, photo_id))
                    if os.path.isfile(save_path_1) and os.path.getsize(save_path_1) > 10 * 1024:
                        print(save_path_1, "【already exists】")
                        continue
                    ret = download(url_path, save_path_1)
                    if ret == 0:
                        print(url_path, save_path_1,
                              main_brand_id, main_brand_name,
                              brand_id, brand_name,
                              series_id, series_name,
                              sell_year,
                              spec_id, spec_name,
                              photo_name)
                    '''
                    if color != -1 and color in [12]:
                        out_dir_2 = os.path.join(args.out_dir, "c_{}".format(color))
                        if not os.path.isdir(out_dir_2):
                            os.makedirs(out_dir_2)
                        save_path_2 = os.path.join(out_dir_2, "{}.{}.{}.jpg".format(sell_year, spec_id, photo_id))
                        if os.path.isfile(save_path_2) and os.path.getsize(save_path_2) > 10 * 1024:
                            print(save_path_2, "【already exists】")
                            continue
                        ret = download(url_path, save_path_2)
                        if ret == 0:
                            print(url_path, save_path_2,
                                  main_brand_id, main_brand_name,
                                  brand_id, brand_name,
                                  series_id, series_name,
                                  sell_year,
                                  spec_id, spec_name,
                                  photo_name)
                    dict_count[property_id] += 1
        print(dict_count['5'], dict_count['6'], dict_count['7'])
                # break
                '''
if __name__ == '__main__':
    main(get_args())


