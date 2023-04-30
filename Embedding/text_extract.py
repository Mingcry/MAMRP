# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
# from transformers import BertTokenizer, BertModel, BertConfig
from pytorch_transformers import BertTokenizer, BertModel, BertConfig
import re


class Text_Extractor(nn.Module):
    def __init__(self, root, dim=100):
        super(Text_Extractor, self).__init__()
        self.dim = dim
        self.tokenizer = BertTokenizer.from_pretrained(root+'/Embedding/Pre_train_model/chinese_L-12_H-768_A-12')
        config = BertConfig.from_pretrained(root+'/Embedding/Pre_train_model/chinese_L-12_H-768_A-12')
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(root+'/Embedding/Pre_train_model/chinese_L-12_H-768_A-12', config=config)
        self.hidden_dim = self.bert.config.hidden_size

        self.fc = nn.Linear(self.hidden_dim, self.dim)
        self.activate = nn.Tanh()

    def handle(self, texts):
        tokens, segments, masks = [], [], []
        for text in texts:
            text = '[CLS]' + text + '[SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)[:512]
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            masks.append([1] * len(indexed_tokens))

        max_len = max([len(a) for a in tokens])  # 最大的句子长度
        for i in range(len(tokens)):
            padding = [0] * (512 - len(tokens[i]))
            tokens[i] += padding
            segments[i] += padding
            masks[i] += padding

        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        masks = torch.tensor(masks)

        return tokens, segments, masks

    def forward(self, text):
        try:
            a = eval(text)
            if isinstance(a, int):
                text = [text]
            elif isinstance(a, list) and a:
                text = a
            else:
                text = [' ']
        except (NameError, SyntaxError):
            text = [text]
        tokens, segments, masks = self.handle(text)
        output = self.bert(tokens, segments, masks)
        sequence, sentence, all_layers = output
        features = torch.mean(torch.mean(all_layers[-2], dim=1), dim=0).unsqueeze(dim=0)
        # return torch.squeeze(features.cpu().detach(), dim=0)
        feat_cls = all_layers[-2].cpu().detach()
        return ((feat_cls)[0][0, :]).unsqueeze(dim=0), feat_cls, torch.squeeze(features.cpu().detach(), dim=0)


if __name__ == '__main__':
    root = 'G:/Pytorch/Movies_Predict/Rating_Prediction'
    text_extract = Text_Extractor(root)

    summary = '近未来，科学家们发现太阳急速衰老膨胀，短时间内包括地球在内的整个太阳系都将被太阳所吞没。为了自救，人类提出一个名为“流浪地球”的大胆计划，即倾全球之力在地球表面建造上万座发动机和转向发动机，推动地球离开太阳系，用2500年的时间奔往另外一个栖息之地。中国航天员刘培强（吴京 饰）在儿子刘启四岁那年前往国际空间站，和国际同侪肩负起领航者的重任。转眼刘启（屈楚萧 饰）长大，他带着妹妹朵朵（赵今麦 饰）偷偷跑到地表，偷开外公韩子昂（吴孟达 饰）的运输车，结果不仅遭到逮捕，还遭遇了全球发动机停摆的事件。为了修好发动机，阻止地球坠入木星，全球开始展开饱和式营救，连刘启他们的车也被强征加入。在与时间赛跑的过程中，无数的人前仆后继，奋不顾身，只为延续百代子孙生存的希望……'
    print(len(summary))

    output = text_extract(summary)[0]
    print(output)
    print(output.shape)






