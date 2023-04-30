import torch
import torch.nn as nn
import math
from GCN.GCN_Net import *
import numpy as np


class MRP(nn.Module):
    def __init__(self, in_dim, gcn_dropout=0.1, out_dim=6, gcn_norm=False, embedding_dim=64, gcn_layers=3):
        super(MRP, self).__init__()
        # self.info_fc = nn.Linear(in_dim, embedding_dim)
        self.text_fc = nn.Linear(in_dim, embedding_dim)
        self.image_fc = nn.Linear(in_dim, embedding_dim)
        self.director_embedding = nn.Embedding(120, embedding_dim)
        self.weight = torch.nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.weight1= torch.nn.Parameter(torch.Tensor([1/3, 1/3, 1/3]))

        self.query = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

        self.FeatFc = nn.Sequential(nn.Linear(in_dim, embedding_dim))
        self.gcn = GCN(in_channels=embedding_dim, hidden_channels1=embedding_dim, hidden_channels2=embedding_dim, out_channels=embedding_dim, dropout=gcn_dropout, need_norm=gcn_norm, n_layers=gcn_layers)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.5))
        self.classifer = nn.Linear(embedding_dim, out_dim)
        self.reg = nn.Linear(embedding_dim, 1)

    def forward(self, id, edge_index, x, index, info_index, text_index, image_index):
        director = self.director_embedding(id)
        info, text, image = x[info_index], x[text_index], x[image_index]
        info, text, image = self.text_fc(info), self.image_fc(text), self.image_fc(image)
        # info, text = self.info_fc(info), self.text_fc(text)

        modals = self.weight[0] * info + self.weight[1] * text
        # modals = self.weight1[0] * info + self.weight1[1] * text +  self.weight1[2] * image

        x = self.FeatFc(x)
        title = x[index]
        gcn_output = self.gcn(x, edge_index)

        # anchor_emb = x[index]
        anchor_emb = gcn_output[index]
        director = F.log_softmax(director, dim=1)
        title = F.log_softmax(title, dim=1)
        modals = F.log_softmax(modals, dim=1)

        output = anchor_emb + title + director + modals

        output_class = self.classifer(output)
        output_reg = self.reg(output)
        return output_class, output_reg


if __name__ == "__main__":
    x = torch.randn(100, 768)
    x.requires_grad = True
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)  # 邻接表
    model = MRP(in_dim=768)
    index = [1, 2]
    output = model(edge_index, x, index)
    print(output.shape)
    print(output)
