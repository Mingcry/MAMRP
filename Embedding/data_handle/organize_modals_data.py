# -*- coding:utf-8 -*-
import os
from os.path import join
import re
import csv
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码


def inverse(d):
    res = {}
    for item in d:
        res[d[item]] = item
    return res

root = 'G:/Pytorch/Movies_Predict/Rating_Prediction/MMMD'
root_save = 'Rating_Prediction/MMMD'
test_path = './new_data/test.txt'
movies_list_path = './new_data/movie_info.csv'
info = {
        '电影名': 0, '电影网址': 1, '导演': 2, '编剧': 3, '主演': 4,
        '评分': 5, '评分人数': 6, '评分额外信息': 7,
        '类型': 8, '制片国家/地区': 9, '语言': 10, '上映时间': 11, '时长': 12,
        '剧情简介': 13, '获奖情况': 14, '热门评论': 15,
        '海报链接': 16, '海报保存路径': 17, '剧照链接': 18, '剧照保存路径': 19, '视频链接': 20, '视频保存路径': 21
    }
ids = inverse(info)

wjj_list = ['data', 'data1', 'data2', 'data3']
e = 3
file_list = os.listdir(root + '/' + wjj_list[e] + '/movie_info')
print(file_list)
for index, file in enumerate(file_list):  # 遍历所有导演的电影信息文件
    # print(file)
    path = root + '/' + wjj_list[e] + '/movie_info/' + file
    dir_name = './new_data/train'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    add = 0
    if e == 1:
        add += 45
    elif e == 2:
        add += 79
    elif e == 3:
        add += 100
    save_path = dir_name + '/' + str(index+add) + '.txt'  # 保存路径

    # print(path)

    data = []  # 文件信息列表
    with open(path, 'r', encoding='utf-8') as f:
        csv_read = csv.reader(f)
        for i, row in enumerate(csv_read):
            if i:
                data.append(row)
    num_movie = len(data)  # 导演的电影个数
    # print('num:', num_movie)

    director_index = {}  # 电影索引存储
    # root
    director = data[0][info['导演']]
    # print(director)
    with open(save_path, 'a+', encoding='utf-8') as f:
        f.write(director + '\n')
        f.write('序号' + '\t' + '评分' + '\t' + '电影名' + '\t' + '电影信息' + '\n')

    k = 0  # 有效电影个数
    for i in range(num_movie):  # 遍历某个导演的所有电影
        movie = data[i]
        for j, item in enumerate(movie):
            if item == 'INF':
                movie[j] = '暂无'
            if j < 16:
                movie[j] = re.sub(' ', '', movie[j].strip('\n'))

        valid = True
        information = []  # 电影各模态信息结点

        # level1
        name = movie[info['电影名']]
        # print(name)

        # level2 text-1
        base = []
        actor = '主演是'
        if movie[info['主演']] != '暂无':
            # print(director, name)
            # print(director, name, movie[info['主演']])
            actor += ','.join(eval(movie[info['主演']]))
        else:
            actor += '暂无'
        base.append(actor)
        writer = '编剧是' + re.sub('/', '、', movie[info['编剧']])
        base.append(writer)
        for i in range(8, 13):
            base.append(ids[i] + '是' + re.sub('/', '、', movie[i]))
        base_info = '。'.join(base) + '。'
        # print(base_info)
        information.append((len(information), base_info))

        # level2 text-2
        introduction = 'None'
        if movie[info['剧情简介']] != '暂无':
            introduction = '剧情简介是' + movie[info['剧情简介']]
        else:
            valid = False
        # print(introduction)
        information.append((len(information), introduction))

        # level2 image1
        image_path = movie[info['海报保存路径']]
        if image_path[:2] == './':
            image_path = image_path[2:]
        if image_path[:5] != wjj_list[e]:
            image_path = wjj_list[e] + image_path[4:]
        image_path1 = join(root, image_path)
        image = os.listdir(image_path1)
        if '海报1.jpg' in image:
            image = join(root_save, image_path) + '/' + '海报1.jpg'
        else:
            image = 'None'
        # print(image)
        information.append((len(information), image))

        # level2 image2
        image_path = movie[info['剧照保存路径']]
        if image_path[:2] == './':
            image_path = image_path[2:]
        if image_path[:5] != wjj_list[e]:
            image_path = wjj_list[e] + image_path[4:]
        image_path1 = join(root, image_path)
        image = os.listdir(image_path1)
        if '剧照1.jpg' in image:
            image = join(root_save, image_path) + '/' + '剧照1.jpg'
        else:
            image = 'None'
        # print(image)
        information.append((len(information), image))

        # level2 video
        video_path = movie[info['视频保存路径']]
        if video_path[:2] == './':
            video_path = video_path[2:]
        if video_path[:5] != wjj_list[e]:
            video_path = wjj_list[e] + video_path[4:]
        video_path1 = join(root, video_path)
        video = os.listdir(video_path1)
        if video:
            video = join(root_save, video_path) + '/' + '视频1.mp4'
        else:
            video = 'None'
        # print(video)
        information.append((len(information), video))

        if valid:
            label = movie[info['评分']]
            with open(save_path, 'a+', encoding='utf-8') as f:
                f.write(str(k) + '\t' + label + '\t' + name + '\t' + str(information) + '\n')
            if not os.path.exists(movies_list_path):
                with open(movies_list_path, 'a+', encoding='utf-8', newline='') as csvfile:
                    csv_write = csv.writer(csvfile)
                    csv_write.writerow(
                        ['电影名', '电影网址', '导演', '编剧', '主演',
                         '评分', '评分人数', '评分额外信息',
                         '类型', '制片国家/地区', '语言', '上映时间', '时长',
                         '剧情简介', '获奖情况', '热门评论',
                         '海报链接', '海报保存路径', '剧照链接', '剧照保存路径', '视频链接', '视频保存路径'
                         ])
            with open(movies_list_path, 'a+', encoding='utf-8', newline='') as csvfile:
                csv_write = csv.writer(csvfile)
                csv_write.writerow(movie)
                csvfile.flush()
            k += 1




