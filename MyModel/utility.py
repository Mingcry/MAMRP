import torch
from os.path import join
from Embedding.text_extract import *
from Embedding.image_extract import *
from Embedding.video_extract import *
import numpy as np
load_root = 'G:/Pytorch/Movies_Predict'
root = join(load_root, 'Rating_Prediction')
text_extract = Text_Extractor(root)


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


def get_adj_x(root, path, in_dim=768, trian=True):
    with open(path, 'r', encoding='utf-8') as f:
        data = [item.strip('\n') for item in f.readlines()]
    director = data[0]  # 导演
    nn = len(data[2:])
    test_num = max(1, int(nn*0.1))
    if trian:
        data = data[2:-test_num]  # 原信息
    else:
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
    index_grad = [_ for _ in range(0, 1 + num_movie)]
    index_train = [_ for _ in range(1, 1+num_movie)]
    index_test = [_ for _ in range(1, 1 + num_movie)]
    index_test = index_test[-test_num:]
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
            if node[1] != 'None' and j == 4:  # 视频信息
                video[k] = node[1]
                k += 1
                node_num += 1
        text_nodes.append(text)
        image_nodes.append(image)
        video_nodes.append(video)
    x = torch.zeros(k, in_dim)
    # x[0, :] = text_extract(director)[0]
    x[0, :] = torch.load(join(root, director))
    A_0, A_1 = [0]*num_movie, [i for i in range(1, num_movie+1)]
    for i in range(len(name_movie)):
        # x[i+1, :] = text_extract(name_movie[i])[0]
        x[i + 1, :] = torch.load(join(root, name_movie[i]))
        for item in text_nodes[i].items():
            # x[item[0], :] = text_extract(item[1])[0]
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
        for item in image_nodes[i].items():
            # x[item[0], :] = Image_Extractor(join(load_root, item[1]))[1]
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
        for item in video_nodes[i].items():
            # x[item[0], :] = Video_Extractor(root, join(load_root, item[1]))
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
    assert len(A_0) == len(A_1)
    label = [get_label(int(i*10)) for i in label]
    if not trian:
        label = label[-test_num:]
    label = torch.LongTensor(np.array(label))
    edge_index = np.concatenate((np.expand_dims(np.array(A_0), axis=0), np.expand_dims(np.array(A_1), axis=0)), axis=0)
    return torch.LongTensor(edge_index), x, index_train, index_test, label, index_grad


def get_adj_x_save(root, path, in_dim=768, trian=True):
    with open(path, 'r', encoding='utf-8') as f:
        data = [item.strip('\n') for item in f.readlines()]
    director = data[0]  # 导演
    nn = len(data[2:])
    test_num = max(1, int(nn*0.1))
    if trian:
        data = data[2:-test_num]  # 原信息
    else:
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
    index_train = [_ for _ in range(1, 1+num_movie)]
    index_test = [_ for _ in range(1, 1 + num_movie)]
    index_test = index_test[-test_num:]
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
            if node[1] != 'None' and j == 4:  # 视频信息
                video[k] = node[1]
                k += 1
                node_num += 1
        text_nodes.append(text)
        image_nodes.append(image)
        video_nodes.append(video)
    x = torch.zeros(k, in_dim)
    # x[0, :] = text_extract(director)[0]
    x[0, :] = torch.load(join(root, director))
    A_0, A_1 = [0]*num_movie, [i for i in range(1, num_movie+1)]
    B_0, A_1 = [0] * num_movie, [i for i in range(1, num_movie + 1)]
    for i in range(len(name_movie)):
        # x[i+1, :] = text_extract(name_movie[i])[0]
        x[i + 1, :] = torch.load(join(root, name_movie[i]))
        for item in text_nodes[i].items():
            # x[item[0], :] = text_extract(item[1])[0]
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
        for item in image_nodes[i].items():
            # x[item[0], :] = Image_Extractor(join(load_root, item[1]))[1]
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
        for item in video_nodes[i].items():
            # x[item[0], :] = Video_Extractor(root, join(load_root, item[1]))
            x[item[0], :] = torch.load(join(root, item[1]))
            A_0.append(i + 1)
            A_1.append(item[0])
    assert len(A_0) == len(A_1)
    label = [get_label(int(i*10)) for i in label]
    if not trian:
        label = label[-test_num:]
    label = torch.LongTensor(np.array(label))
    edge_index = np.concatenate((np.expand_dims(np.array(A_0), axis=0), np.expand_dims(np.array(A_1), axis=0)), axis=0)
    return torch.LongTensor(edge_index), x, index_train, index_test, label

def get_edge_index(num_movie, edge, sam_num, train=True):
    A_0, A_1 = [0] * num_movie, [i for i in range(1, num_movie + 1)]
    info_index = []
    text_index = []
    image_index = []
    for item in edge.items():
        if item[0] == 0:
            continue
        else:
            info_index.append(item[1][0])
            text_index.append(item[1][1])
            image_index.append(item[1][2])
            for i in range(len(item[1])):
                A_0.append(item[0])
                A_1.append(item[1][i])
    edge_index = np.concatenate((np.expand_dims(np.array(A_0), axis=0), np.expand_dims(np.array(A_1), axis=0)), axis=0)
    edge_index = torch.LongTensor(edge_index)
    if train:
        info_index = info_index[:sam_num]
        text_index = text_index[:sam_num]
        image_index = image_index[:sam_num]
    else:
        info_index = info_index[-(num_movie - sam_num):]
        text_index = text_index[-(num_movie - sam_num):]
        image_index = image_index[-(num_movie - sam_num):]
    return edge_index, info_index, text_index, image_index

if __name__ == "__main__":
    root = 'G:/Pytorch/Movies_Predict/Rating_Prediction'
    paths = os.listdir(join(root, 'Embedding/data_handle/feat_data'))
    paths = sorted(paths, key=lambda x: int(x.split('.')[0]))
    paths = [join(join(root, 'Embedding/data_handle/feat_data'), i) for i in paths]
    print(paths)
    d = {0: [1, 2, 3], 1: [4, 5, 6, 7, 8], 2: [9, 10, 11, 12, 13], 3: [14, 15, 16, 17, 18]}
    num = 3
    print(get_edge_index(num, d))
    # for path in paths:
    #     print(path)
    #     ids = path.split('\\')[-1]
    #     id = int(ids.split('.')[0])
    #     feats_dict = {}
    #     edge_index, x, index_train, _, label = get_adj_x_save(root, path, trian=True)
    #     feats_dict['train'] = {'edge_index': edge_index, 'x': x, 'index_emd': index_train, 'label': label}
    #
    #     edge_index, x, _, index_test, label = get_adj_x_save(root, path, trian=False)
    #     feats_dict['test'] = {'edge_index': edge_index, 'x': x, 'index_emd': index_test, 'label': label}
    #
    #     feat_path = join(root, f'Embedding/data_handle/feat_matrix_data/{id}.pt')
    #     torch.save(feats_dict, feat_path)
    #     d = torch.load(feat_path)
    #     print(d['train']['x'].shape)
    #     print(d['test']['x'].shape)
    # path = join(root, 'Embedding/data_handle/feat_data/0.txt')
    # feats_dict = {}
    #
    # edge_index, x, index_train, _, label, index_grad = get_adj_x_save(root, path, trian=True)
    # print(x)
    # print(edge_index)
    # print(x.shape)
    # print(edge_index.shape)
    # print(label)
    # print(index_train)
    # feats_dict['train'] = {'edge_index': edge_index, 'x': x, 'index_emd': index_train, 'label': label, 'index_grad': index_grad}
    #
    # edge_index, x, index_train, index_test, label, _ = get_adj_x_save(root, path, trian=False)
    # print(x)
    # print(edge_index)
    # print(x.shape)
    # print(edge_index.shape)
    # print(label)
    # print(index_test)
    # feats_dict['test'] = {'edge_index': edge_index, 'x': x, 'index_emd': index_test, 'label': label}
    #
    # feat_path = join(root, 'Embedding/data_handle/feat_matrix_data/0.pt')
    # torch.save(feats_dict, feat_path)
    # d = torch.load(feat_path)
    # print(d)
    # print(len(d), type(d))
    # print(d['train']['x'].shape)
    # print(d['test']['x'].shape)

