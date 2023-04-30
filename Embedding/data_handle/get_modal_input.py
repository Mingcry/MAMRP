import numpy as np
import pandas as pd
import torch
from os.path import join
import random
import joblib
from random import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import lightgbm
import xgboost as xgb
import os
import warnings
warnings.filterwarnings("ignore")

j = 5
epochs = 1
training_state = []

root = 'G:/Pytorch/Movies_Predict/Rating_Prediction'
train_path = os.path.join(root, 'Embedding/data_handle/new_data/baseline_data_input.csv')
director_pca = joblib.load(root+'/'+'Embedding/data_handle/new_data/pca_director.pkl')
title_pca = joblib.load(root+'/'+'Embedding/data_handle/new_data/pca_title.pkl')
info_pca = joblib.load(root+'/'+'Embedding/data_handle/new_data/pca_info.pkl')
text_pca = joblib.load(root+'/'+'Embedding/data_handle/new_data/pca_text.pkl')
image_pca = joblib.load(root+'/'+'Embedding/data_handle/new_data/pca_image.pkl')


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


def read_data(path):
    data = pd.read_csv(path)
    n = data.shape[0]


    num = 1799-169
    x_tr, y_tr = x_res[:num], y_res[:num]
    x_te, y_te = x_res[num:], y_res[num:]

    return x_tr, y_tr, x_te, y_te


# input_get = torch.load('./data/input/modals_not/data_dict(169shuffle).pt')
# input_get = torch.load('./data/input/modals/data_dict(169shuffle).pt')
x_train, y_train, x_test, y_test = read_data(train_path)
# x_train, y_train, x_test, y_test = input_get['x_train'], input_get['y_train'], input_get['x_test'], input_get['y_test']
# index = [258, 514, 770]
data_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
feat_path = './data/input/modals/data_dict(169shuffle).pt'
torch.save(data_dict, feat_path)