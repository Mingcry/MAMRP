import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--state_path', nargs='?', default='MyModel/result/train_state.csv',
                        help='state_path')
    parser.add_argument('--state_pig_path', nargs='?', default='MyModel/result/train_state.png',
                        help='state_pig_path')
    parser.add_argument('--model_path', nargs='?', default='MyModel/model_save/best_bingo_model.pth',
                        help='model_path')
    parser.add_argument('--root', nargs='?', default='G:/Pytorch/Movies_Predict/Rating_Prediction',
                        help='root_path')
    parser.add_argument('--train_path', nargs='?', default="['Embedding/data_handle/feat_dict_data', 'Embedding/data_handle/feat_dict_data(TextImage)', 'Embedding/data_handle/feat_dict_data(onlyText)']",
                        help='train_path')
    parser.add_argument('--train_path_select', nargs='?', default=0,
                        help='train_path_select')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed')
    parser.add_argument('--gcn_norm', type=bool, default=False,
                        help='gcn_norm?')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--print_step', type=int, default=30,
                        help='Print Train Info.')
    parser.add_argument('--lr', type=float, default=0.015,
                        help='Learning rate.')
    parser.add_argument('--lambda_loss', type=float, default=1e-2,
                        help='lambda_loss')
    parser.add_argument('--in_dim', type=int, default=768,
                        help='Embedding dim.')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='embedding dim.')
    parser.add_argument('--gcn_layers', type=int, default=3,
                        help='gcn_layers.')
    parser.add_argument('--gcn_dropout', nargs='?', default=0.1,
                        help='gcn dropout (i.e., 1-dropout_ratio)')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU id')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-4,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='') 
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of item graph conv layers')  
    parser.add_argument('--dropout', nargs='?', default='[0.1, 0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')
    parser.add_argument('--loss_ratio', type=float, default=0.03, help='Control the effect of the contrastive auxiliary task')        
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

            

    return parser.parse_args()
