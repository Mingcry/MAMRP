import torch
from os.path import join
import os
import numpy as np
load_root = 'G:/Pytorch/Movies_Predict'
root = join(load_root, 'Rating_Prediction')


def get_label(score):
    linjie = [65, 71, 75, 79, 84]
    y = 0
    for i, item in enumerate(linjie):
        if score < item and i < len(linjie)-1:
            y = i
            break
        if score < item and i == len(linjie)-1:
            y = i
            break
        if score >= item and i == len(linjie)-1:
            y = i+1
            break
    return y


def get_adj_x_save(root, path, in_dim=768, trian=True):
    with open(path, 'r', encoding='utf-8') as f:
        data = [item.strip('\n') for item in f.readlines()]
    director = data[0]  # 导演
    nn = len(data[2:])
    test_num = max(1, int(nn*0.1))
    data = data[2:]  # 原信息
    num_movie = len(data)  # 电影数量
    name_movie = []  # 电影名
    label = []  # 评分
    info = []  # 信息
    for i in range(len(data)):
        movie = data[i].split('\t')[1:]
        label.append(float(movie[0]))
        name_movie.append(movie[1])
        info.append(eval(movie[2]))  # 里面是一个列表
    k = 1 + num_movie  # 存第三层结点
    node_num = 1 + num_movie  # 结点个数
    text_nodes = []
    image_nodes = []
    video_nodes = []
    index = [_ for _ in range(1, 1 + num_movie)]
    index_train = index[:-test_num]
    index_test = index[-test_num:]
    edge = {}
    edge[0] = [_ for _ in range(1, 1 + num_movie)]
    for item in info:
        text = {}
        image = {}
        video = {}
        for j, node in enumerate(item):
            if node[1] != 'None' and j < 2:  # 文本信息j=0,1
                text[k] = node[1]
                k += 1
                node_num += 1
            if node[1] != 'None' and (2 <= j <= 3):  # 图片信息
                image[k] = node[1]
                k += 1
                node_num += 1
            # if node[1] != 'None' and j == 4:  # 视频信息
            #     video[k] = node[1]
            #     k += 1
            #     node_num += 1
        text_nodes.append(text)
        image_nodes.append(image)
        # video_nodes.append(video)
    x = torch.zeros(k, in_dim)
    x[0, :] = torch.load(join(root, director))
    A_0, A_1 = [0]*num_movie, [i for i in range(1, num_movie+1)]
    B_0, A_1 = [0] * num_movie, [i for i in range(1, num_movie + 1)]
    for i in range(len(name_movie)):
        if i+1 not in edge:
            edge[i+1] = []
        x[i + 1, :] = torch.load(join(root, name_movie[i]))
        for item in text_nodes[i].items():
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
            edge[i + 1].append(item[0])
        for item in image_nodes[i].items():
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
            edge[i + 1].append(item[0])
        # for item in video_nodes[i].items():
        #     x[item[0], :] = torch.load(join(root, item[1]))
        #     A_0.append(i + 1)
        #     A_1.append(item[0])
        #     edge[i + 1].append(item[0])
    assert len(A_0) == len(A_1)
    label_class = [get_label(int(i*10)) for i in label]
    label_reg = label
    label_train = torch.LongTensor(np.array(label_class[:-test_num]))
    label_test = torch.LongTensor(np.array(label_class[-test_num:]))
    label_train_reg = torch.FloatTensor(np.array(label_reg[:-test_num]))
    label_test_reg = torch.FloatTensor(np.array(label_reg[-test_num:]))
    edge_index = np.concatenate((np.expand_dims(np.array(A_0), axis=0), np.expand_dims(np.array(A_1), axis=0)), axis=0)
    edge_index = torch.LongTensor(edge_index)
    return num_movie, edge, x, index_train, index_test, label_train, label_test, label_train_reg, label_test_reg

if __name__ == "__main__":
    root = 'G:/Pytorch/Movies_Predict/Rating_Prediction'
    paths = os.listdir(join(root, 'Embedding/data_handle/feat_data'))
    paths = sorted(paths, key=lambda x: int(x.split('.')[0]))
    paths = [join(join(root, 'Embedding/data_handle/feat_data'), i) for i in paths]
    print(paths)
    all_num = 0
    for path in paths:
        print(path)
        ids = path.split('\\')[-1]
        id = int(ids.split('.')[0])
        num_movie, edge, x, index_train, index_test, label_train, label_test, label_train_reg, label_test_reg = get_adj_x_save(root, path, trian=True)
        all_num += num_movie
        feats_dict = {'num_movie': num_movie, 'edge': edge, 'x': x, 'index_train': index_train, 'index_test': index_test, 'label_train': label_train, 'label_test': label_test, 'label_train_reg':label_train_reg, 'label_test_reg': label_test_reg}
        feat_path = join(root, f'Embedding/data_handle/feat_dict_data(TextImage)/{id}.pt')
        torch.save(feats_dict, feat_path)
        d = torch.load(feat_path)
        print(d['num_movie'], d['x'].shape)
        # print(d['label_train'])
        # print(d['label_test'])
        # print(d['label_train_reg'])
        # print(d['label_test_reg'])
        # break
    print('all_num=', all_num)




