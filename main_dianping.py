"""
@author:chenyankai
@file:main_movie.py
@time:2020/12/30
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
from src.exp import *
from os.path import join
import sys
from utility.data_loader import data_split

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
ROOT = join(PATH, '../')
sys.path.append(ROOT)

if __name__ == '__main__':
    # dianping
    parser = argparse.ArgumentParser(description='Parse for CG-KGR.')
    parser.add_argument('--data_dir', type=str, default='data/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='dianping', help='select a dataset, e.g., last-fm.')
    parser.add_argument('--kg_file', type=str, default='kg_final.txt', help='select kg file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='select gpu_id')
    parser.add_argument('--node_dim', type=int, default=64, help='the dimension of users, items and entities')
    parser.add_argument('--n_layer', type=int, default=1, help='the number of layers')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size.')
    parser.add_argument('--sample_size', type=int, default=8, help='the size of neighbor samples')
    parser.add_argument('--agg_type', type=str, default='concat', help='specify the type of aggregation for entities from {sum, concat, ngh}')
    parser.add_argument('--repr_type', type=str, default='combine', help='specify the type of creating user-item representative ')
    parser.add_argument('--a', type=float, default=0.4)
    parser.add_argument('--b', type=float, default=0.6)
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--n_epoch', type=int, default=3, help='Number of epoch.')
    parser.add_argument('--seed', type=int, default=2021, help='selected seed for training')
    parser.add_argument('--task', type=str, default='ALL', help='[topk, ctr]')

    args = parser.parse_args()
    saved_dir = 'logs/CG-KGR/{}/Dim{}/'.format(args.data_name, args.node_dim)
    args.saved_dir = saved_dir

    # data_split(args)

    # Exp_all(args)

    