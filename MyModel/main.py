from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from MyModel.utility import *
from MyModel.metrics import *
from MyModel.plot import *
from MyModel.parser import parse_args
from MyModel.model import MRP


args = parse_args()


class Trainer(object):
    def __init__(self, device):
        # argument settings
        self.device = device
        self.lr = args.lr
        self.emb_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.in_dim = args.in_dim
        self.root = args.root
        self.train_path = eval(args.train_path)[args.train_path_select]
        self.gcn_layers = args.gcn_layers

        self.model = MRP(self.in_dim, gcn_dropout=args.gcn_dropout, gcn_norm=args.gcn_norm, embedding_dim=self.emb_dim, gcn_layers=self.gcn_layers)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_add = nn.SmoothL1Loss()
        self.loss_fn_reg = nn.MSELoss()

        self.training_state = []

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 20)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, ):
        t1 = time.time()
        paths = [join(join(self.root, self.train_path), i) for i in os.listdir(join(self.root, self.train_path))]
        loss = 0
        c, c1, ns = 0, 0, 0
        mse_c, mse_log, mae_c, mape_c = 0, 0, 0, 0
        for idx in range(len(paths)):
            self.model.eval()
            # edge_index, x, index_train, index_test, label, index_grad = get_adj_x(self.root, paths[idx], trian=False)
            dict_get = torch.load(paths[idx])
            num_movie, edge, x = dict_get['num_movie'], dict_get['edge'], dict_get['x']
            index_train, index_test, label_train, label_test, label_test_reg = dict_get['index_train'], dict_get['index_test'], dict_get['label_train'], dict_get['label_test'], dict_get['label_test_reg']
            label_test_reg = label_test_reg.view(label_test_reg.shape[0], 1)
            sam_num = len(index_train)
            edge_index, info_index, text_index, image_index = get_edge_index(num_movie, edge, sam_num, train=False)
            ids = [idx]*num_movie
            ids = torch.LongTensor(np.array(ids[:label_test.shape[0]]))
            output, output_reg = self.model(ids, edge_index, x, index_test, info_index, text_index, image_index)
            correct, n = bingo(output, label_test)
            mse_c += float(mse(output_reg, label_test_reg))
            mse_log += float(math.log(mse(output_reg, label_test_reg)))
            mae_c += float(mae(output_reg, label_test_reg))
            mape_c += float(mape(output_reg, label_test_reg))
            c += correct
            c1 += away_1(output, label_test)
            ns += n
            loss_batch = self.loss_fn(output, label_test)
            loss += float(loss_batch)
        loss /= len(paths)
        bingo_acc = c / ns
        away_acc = c1 / ns
        mse_c /= ns
        mse_log /= ns
        mae_c /= ns
        mape_c /= ns
        print('[%.1fs], test_loss==[%.5f], samples==[%d], acc==[[%.5f] [%.5f]] mae==[%.5f]' % (time.time()-t1, loss, ns, bingo_acc, away_acc, mae_c))
        return loss, bingo_acc, away_acc, mse_c, mse_log, mae_c, mape_c

    def train(self, ):
        training_time_list = []
        best_bingo = 0
        for epoch in (range(args.epoch)):
            t1 = time.time()
            loss, total_correct1, total_correct2, sameples = 0., 0., 0., 0.
            paths = [join(join(self.root, self.train_path), i) for i in os.listdir(join(self.root, self.train_path))]
            for idx in range(len(paths)):
                self.model.train()
                self.optimizer.zero_grad()
                # edge_index, x, index_train, index_test, label, index_grad = get_adj_x(self.root, paths[idx], trian=True)
                dict_get = torch.load(paths[idx])
                num_movie, edge, x = dict_get['num_movie'], dict_get['edge'], dict_get['x']
                index_train, index_test, label_train, label_test, label_train_reg = dict_get['index_train'], dict_get['index_test'], dict_get['label_train'], dict_get['label_test'], dict_get['label_train_reg']
                label_train_reg = label_train_reg.view(label_train_reg.shape[0], 1)
                sam_num = len(index_train)
                edge_index, info_index, text_index, image_index = get_edge_index(num_movie, edge, sam_num, train=True)
                ids = [idx] * num_movie
                ids = torch.LongTensor(np.array(ids[:label_train.shape[0]]))
                output, output_reg = self.model(ids, edge_index, x, index_train, info_index, text_index, image_index)
                correct, n = bingo(output, label_train)
                total_correct1 += correct
                total_correct2 += away_1(output, label_train)
                sameples += n
                loss_batch = self.loss_fn(output, label_train)
                loss_reg = self.loss_fn_add(output_reg, label_train_reg)
                loss_reg_only = self.loss_fn_reg(output_reg, label_train_reg)

                # loss_ = loss_batch + args.lambda_loss * loss_reg

                loss_ = loss_batch
                # loss_ = args.lambda_loss * loss_batch + loss_reg
                # loss_ = loss_reg_only

                loss_id = loss_batch
                if idx % args.print_step == 0:
                    print(f'[epoch-{epoch}:{idx}/{len(paths)} loss_batch:{float(loss_id):.6}+{float(loss_reg):.6}]')
                loss_.backward()
                self.optimizer.step()
                loss += float(loss_id)

            self.lr_scheduler.step()
            loss /= len(paths)
            bingo_acc = total_correct1 / sameples
            away_acc = total_correct2 / sameples

            if math.isnan(loss):
                print('ERROR: loss is nan.')
                sys.exit()

            perf_str = 'Epoch %d [%.1fs]: train_loss==[%.5f], samples=[%d], acc==[[%.5f] [%.5f]]' % (
                epoch, time.time() - t1, loss, sameples, bingo_acc, away_acc)
            training_time_list.append(time.time() - t1)
            print(perf_str)
            t_loss, t_bingo, t_away, mse_c, mse_log, mae_c, mape_c = self.test()
            if t_bingo > best_bingo:
                best_bingo = t_bingo
                torch.save(self.model.state_dict(), join(self.root, args.model_path))
                print('Best Bingo Achievement!')
            print('')
            self.training_state.append(
                {
                    'epoch': epoch + 1,
                    'train_loss': float(loss),
                    'test_loss': float(t_loss),
                    'train_bingo': float(bingo_acc),
                    'test_bingo': float(t_bingo),
                    'test_away': float(t_away),
                    'learnRate': self.lr,
                    'mse' : mse_c,
                    'mse_log': mse_log,
                    'mae': mae_c,
                    'mape': mape_c
                }
            )
        df_state = pd.DataFrame(self.training_state)
        csv_path = join(self.root, args.state_path)
        df_state.to_csv(csv_path)
        pig_path = join(self.root, args.state_pig_path)
        plot_main(csv_path, pig_path)
        result = pd.read_csv(csv_path)
        bingo_all = list(result['test_bingo'])
        away_all = list(result['test_away'])
        mse_all = list(result['mse'])
        mse_log_all = list(result['mse_log'])
        mae_all = list(result['mae'])
        mape_all = list(result['mape'])
        record = sorted(list(zip(bingo_all, away_all)), key=lambda xx: xx[0], reverse = True)
        print(f'Result:[bingo=={record[0][0]:.6}] [1-away=={record[0][1]:.6}, {max(away_all):.6}]')
        print(f'Result:[bingo=={record[1][0]:.6}] [1-away=={record[1][1]:.6}, {max(away_all):.6}]')
        print(f'Result:[bingo=={record[2][0]:.6}] [1-away=={record[2][1]:.6}, {max(away_all):.6}]')
        print(f'Result:Reg [mse=={min(mse_all):.2} mse_log=={min(mse_log_all):.2}, mae=={min(mae_all):.2}, mape=={min(mape_all):.6}]')


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    trainer = Trainer(device)
    trainer.train()

