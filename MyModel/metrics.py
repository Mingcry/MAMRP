# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
import pandas as pd
import time
import numpy as np


def cal_time(t1, t2):
    t = (t2 - t1).seconds
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    h = t % 60
    return str(h) + ' h ' + str(m) + ' m ' + str(s) + ' s '


def mae(y_true, y_pred):
    return torch.sum(torch.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return torch.sum(torch.multiply(y_true - y_pred, y_true - y_pred))


def mape(y_true, y_pred):
    return torch.sum(torch.abs((y_pred - y_true) / y_true))


def smape(y_true, y_pred):
    return 2.0 * torch.sum(torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true))) * 100


def bingo(output, labels):
    output = F.softmax(output, dim=1)
    a = torch.sum(torch.argmax(output, dim=1) == labels)
    return a, output.shape[0]


def away_1(output, labels):
    labels_0 = labels - 1
    labels_1 = labels + 1
    a = torch.sum(torch.argmax(output, dim=1) == labels)
    a += torch.sum(torch.argmax(output, dim=1) == labels_0) + torch.sum(torch.argmax(output, dim=1) == labels_1)
    return a

