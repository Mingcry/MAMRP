# -*- coding:utf-8 -*-
# @file  :split_data.py
# @time  :2023/03/21
# @author:qmf
import os
from os.path import join
import re
import csv
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码


root = 'G:/Pytorch/Movies_Predict/Rating_Prediction/MMMD'
train_path = './new_data/train'
test_path = './new_data/test.txt'
if not os.path.exists(test_path):
    with open(test_path, 'a+', encoding='utf-8') as f:
        f.write(str('导演id') + '\t' + '标签' + '\t' + '电影名' + '\t' + '电影信息' + '\n')
director_list = os.listdir(train_path)
director_list = sorted(director_list, key=lambda x: int(x.split('.')[0]))
for item in director_list:
    id = item.split('.')[0]
    new = [id]
    with open(train_path+'/'+item, 'r', encoding='utf-8') as f:
        read = f.readlines()[-1]
        new.extend(read.split('\t')[1:])
    with open(test_path, 'a+', encoding='utf-8') as f:
        f.write('\t'.join(new))

