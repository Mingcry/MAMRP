# -*- coding:utf-8 -*-
# @file  :GCN_Net.py
# @time  :2023/03/21
# @author:qmf
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数的初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def adjacency_matrix(self, edge_index, num_nodes=None, is_lastLayer=False):
        # 将稀疏的邻接矩阵转换为稠密矩阵
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
        row, col = edge_index
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj[row, col] = 1
        # adj[col, row] = 1
        if is_lastLayer:
            adj[col, row] = 1
        adj = adj.to_sparse()
        dense_adj = adj.to_dense()
        identity = torch.eye(adj.size(0))
        dense_adj = torch.add(dense_adj, identity)
        adj = dense_adj.to_sparse()
        return adj

    def forward(self, input, edge_index, need_norm, is_lastLayer=False):
        adj = self.adjacency_matrix(edge_index, input.size(0), is_lastLayer)
        # 对邻接矩阵进行规范化
        if need_norm:
            deg = torch.sparse.sum(adj, dim=1).to_dense()  # 计算每个节点的度数
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 处理度数为 0 的节点
            norm = torch.diag(deg_inv_sqrt).mm(adj.to_dense()).mm(torch.diag(deg_inv_sqrt))
            adj = norm.to_sparse()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, dropout=0.1, need_norm=False, n_layers=3):
        super(GCN, self).__init__()
        # self.layers = nn.ModuleList([GraphConvolution(in_channels, hidden_channels1) for _ in range(n_layers)])
        self.conv1 = GraphConvolution(in_channels, hidden_channels1)
        self.conv2 = GraphConvolution(hidden_channels1, hidden_channels2)
        self.conv3 = GraphConvolution(hidden_channels2, out_channels)
        # self.conv4 = GraphConvolution(hidden_channels2, out_channels)
        # self.conv5 = GraphConvolution(hidden_channels2, out_channels)
        # self.conv6 = GraphConvolution(hidden_channels2, out_channels)
        self.dropout = dropout
        self.need_norm = need_norm

    def forward(self, x, edge_index):
        # edge_index: 邻接矩阵（2 x m），每列表示一条边的两个端点的索引
        # for layer in self.layers:
        #     x = layer(x, edge_index, self.need_norm)
        #     x = F.relu(x)
        #     x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index, self.need_norm)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index, self.need_norm)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv3(x, edge_index, self.need_norm, is_lastLayer=True)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.log_softmax(x, dim=1)
        # x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    x = torch.randn(10, 16)  # 10 个节点，每个节点特征为 16 维
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)  # 邻接表
    model = GCN(in_channels=16, hidden_channels1=32, hidden_channels2=32, out_channels=16)
    output = model(x, edge_index)
    print(output.shape)

