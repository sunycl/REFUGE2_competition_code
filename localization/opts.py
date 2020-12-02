import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--trainset_path', default='pancreas', type=str,
                        help='Trainset directory path of data')
    parser.add_argument('-ts', '--testset_path', default='', type=str,
                        help='Testset directory path of data')
    parser.add_argument('-rs', '--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='Initial learning rate')
    parser.add_argument('--input_list', default='128,128,128', help='List of input')
    parser.add_argument('--weight_decay', default=2e-6, type=float, help='Weight Decay')
    parser.add_argument('--n_epochs', default=35, type=int, help='Number of total epochs to run')
    parser.add_argument('--lr_decay_epochs', default=200, type=int, help='Number of epochs to decay LR')
    parser.add_argument('--fold', default=4, type=int, help='Totaly 5 flod')
    parser.add_argument('--seed', default=1234, type=int, help='Manually set random seed')
    parser.add_argument('--vis', default=False, type=bool, help='Visualization')
    parser.add_argument('--is_pretrain', default=False, type=int, help='pretrained')
    parser.add_argument('--pretrain_path', default='checkpoint/latest_net_G.pth', type=str, help='net_G_229_0.00000.pkl')
    parser.add_argument('-m', '--model', default='get_efficientunet_b5', type=str, help='Model Architect')
    parser.add_argument('-scale', '--feature_scale', default=4, type=float, help='Model feature_scale')
    parser.add_argument('-id', '--gpuid', default=1, type=int, help='Which gpu to use')
    parser.add_argument('-bs', '--batchsize', default=16, type=int, help='batchsize')
    args = parser.parse_args()
    return args
