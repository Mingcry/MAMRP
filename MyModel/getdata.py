import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from random import *


def get_label(score):
    y = 0
    if score in range(65, 71):
        y = 1
    elif score in range(71, 75):
        y = 2
    elif score in range(75, 79):
        y = 3
    elif score in range(79, 84):
        y = 4
    elif score in range(84, 100):
        y = 5
    return y


def get_dataset_train(path_train, path_test):
    with open(path_train, 'r') as f:
        info_train = [item.strip('\n') for item in f.readlines()[1:]]
    res_train = []
    for i in range(len(info_train)):
        movie = info_train[i].split('\t')[1:]
        if movie:
            y = get_label(int(float(movie[0])*10))
            # y = float(movie[0])
            new_movie = [y, eval(movie[1]), float(movie[0])]
            res_train.append(new_movie)
    shuffle(res_train)

    with open(path_test, 'r') as f:
        info_test = [item.strip('\n') for item in f.readlines()[1:]]
    res_test = []
    for i in range(len(info_test)):
        movie = info_test[i].split('\t')[1:]
        if movie:
            y = get_label(int(float(movie[0])*10))
            # y = float(movie[0])
            new_movie = [y, eval(movie[1]), float(movie[0])]
            res_test.append(new_movie)
    shuffle(res_test)

    return res_train, res_test


def get_dataset_init(path, radio):
    with open(path, 'r') as f:
        info = [item.strip('\n') for item in f.readlines()[1:]]
    res = []
    for i in range(len(info)):
        movie = info[i].split('\t')[1:]
        if movie:
            y = get_label(int(float(movie[0])*10))
            # y = float(movie[0])
            new_movie = [y, eval(movie[1]), float(movie[0])]
            res.append(new_movie)
    shuffle(res)
    num = int(radio*len(res))
    return res[:num], res[num:]


class LoadData(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, dataset, train_flag=True):
        self.data_info = dataset
        self.train_flag = train_flag

    def __getitem__(self, index):
        res = self.data_info[index]
        label = int(res[0])
        feat = res[1]
        score = res[2]
        return torch.tensor(feat), torch.tensor(label), torch.tensor(score)

    def __len__(self):
        return len(self.data_info)


# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
if __name__ == "__main__":
    batch_size = 16
    train_path = './data/train_data.txt'
    p = 0.2
    train, test = get_dataset(train_path, p)
    train_dataset = LoadData(train, False)
    test_dataset = LoadData(test, False)
    print('训练集数据个数：', len(train_dataset))
    print('测试集数据个数：', len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1):
        for i, data in enumerate(train_loader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            feat, labels = data

            # 将这些数据转换成Variable类型
            feat, labels = Variable(feat), Variable(labels)

            # 接下来就是跑模型的环节了，我们这里使用print来代替
            print("epoch：", epoch, "的第", i, "个特征", feat.data.size(),
                  "labels", labels.data.size())

        print('-----------------------------------------------------------------------------')

        for i, data in enumerate(test_loader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            feat, labels = data

            # 将这些数据转换成Variable类型
            feat, labels = Variable(feat), Variable(labels)

            # 接下来就是跑模型的环节了，我们这里使用print来代替
            print("epoch：", epoch, "的第", i, "个特征", feat.data.size(),
                  "labels", labels.data.size())


