# -*- coding:utf-8 -*-
# @file  :get_video_embeddings.py
# @time  :2023/02/25
# @author:qmf
import os
import time
import sys
from os.path import join
from Embedding.text_extract import *
from Embedding.image_extract import *
from Embedding.video_extract import *
import torch
import io
import re

root = 'G:/Pytorch/Movies_Predict/Rating_Prediction'
load_root = 'G:/Pytorch/Movies_Predict'
text_extract = Text_Extractor(root)

# root = 'G:/Pytorch/Movies_Predict/Office_Prediction'
old_data_path = join(root, 'Embedding/data_handle/new_data/train')
embedding_path = join(root, 'Embedding/Embeddings')
directors_list = os.listdir(old_data_path)
directors_list = sorted(directors_list, key=lambda x: int(x.split('.')[0]))
for i, item in enumerate(directors_list):
    if i > 1:
        continue
    id = i
    path = join(old_data_path, item)
    with open(path, 'r', encoding='utf-8') as f:
        reads = f.readlines()
        reads = [j.strip('\n') for j in reads]
    director = reads[0]
    try:
        print(f'==========================================={id}-{director}============================================')
    except:
        print(f'==========================================={id}============================================')
    temp_embedding_path = 'Embedding/Embeddings'
    new_path = join(join(root, 'Embedding/data_handle/feat_data'))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(join(root, temp_embedding_path)):
        os.makedirs(join(root, temp_embedding_path))
    with open(join(new_path, str(id)+'.txt'), 'a+', encoding='utf-8') as f:
        output = text_extract(director)[0]
        # torch.save(output, embedding_path + '/' + director + '.pt')
        f.write(join(temp_embedding_path, director + '.pt')+'\n')
        print('导演名', (torch.load(join(root, join(temp_embedding_path, director + '.pt')))).shape)
        f.write(reads[1]+'\n')
    reads = reads[2:]
    for item in reads:
        movie = item.split('\t')
        new = [movie[0], movie[1]]
        title, info = movie[2], eval(movie[3])
        title = re.sub('[,.:：!?。，@/#*]', '', title)
        try:
            print(f'==============={title}==================')
        except:
            print(f'==============={title[0]}==================')
        movie_path = embedding_path + '/' + title
        # torch.save(text_extract(title)[0], movie_path+'.pt')
        new.append(join(temp_embedding_path, title+'.pt'))
        print('电影名', (torch.load(join(root, join(temp_embedding_path, title+'.pt')))).shape)
        new_temp = []
        for st in info:
            if st[1] == 'None':
                new_temp.append((st[0], 'None'))
                continue
            if st[0] == 0 or st[0] == 1:
                # torch.save(text_extract(st[1])[0], movie_path + '-' + str(st[0]) + '.pt')
                new_temp.append((st[0], join(temp_embedding_path, title + '-' + str(st[0]) + '.pt')))
                print('文本', (torch.load(join(root, join(temp_embedding_path, title + '-' + str(st[0]) + '.pt')))).shape)
            if st[0] == 2 or st[0] == 3:
                # torch.save(Image_Extractor(join(load_root, st[1]))[1], movie_path + '-' + str(st[0]) + '.pt')
                new_temp.append((st[0], join(temp_embedding_path, title + '-' + str(st[0]) + '.pt')))
                print('图片', (torch.load(join(root, join(temp_embedding_path, title + '-' + str(st[0]) + '.pt')))).shape)
            if st[0] == 4:
                # torch.save(Video_Extractor(root, join(load_root, st[1])), movie_path + '-' + str(st[0]) + '.pt')
                new_temp.append((st[0], join(temp_embedding_path, title + '-' + str(st[0]) + '.pt')))
                print('视频', (torch.load(join(root, join(temp_embedding_path, title + '-' + str(st[0]) + '.pt')))).shape)
        new.append(str(new_temp))
        with open(join(new_path, str(id) + '.txt'), 'a+', encoding='utf-8') as f:
            f.write('\t'.join(new) + '\n')

