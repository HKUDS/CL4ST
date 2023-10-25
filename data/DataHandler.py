from scipy import io
import numpy as np
import datetime
from Params import args, logger
import torch
from torch.utils.data import DataLoader
import os
from dgl.dataloading import GraphDataLoader
from data.STGDataset import STGDataset
from utils.util import normalize_dataset, split_data_by_ratio, Add_Window_Horizon,get_adjacency_binary
import matplotlib.pyplot as plt
import seaborn as sns

class DataHandler:
    def __init__(self):
        pass
    
    def get_dataloader(self, normalizer='max01'):
        data = self.load_st_dataset(args.dataset)  # B, N, D
        data, scaler = normalize_dataset(data, normalizer, False)

        tra_path = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon), 'tra_graph_lag{}_hoz{}.mat'.format(args.lag, args.horizon))
        if os.path.exists(tra_path) is False:
            self.get_raw_data(data)
        adj = self.build_adj()
        # adj_hops = self.
        self.eval_adj(adj)
        print(os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon)))
        tra_ds = STGDataset(adj,name='tra_graph_lag{}_hoz{}'.format(args.lag, args.horizon),  raw_dir = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon)))
        val_ds = STGDataset(adj,name='val_graph_lag{}_hoz{}'.format(args.lag, args.horizon),  raw_dir = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon)))
        tst_ds = STGDataset(adj,name='tst_graph_lag{}_hoz{}'.format(args.lag, args.horizon),  raw_dir = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon)))
        tra_loader = GraphDataLoader( tra_ds, batch_size=args.batch_size, drop_last=False, shuffle=True)
        val_loader = GraphDataLoader( val_ds, batch_size=args.batch_size, drop_last=False, shuffle=True)
        tst_loader = GraphDataLoader( tst_ds, batch_size=args.batch_size, drop_last=False, shuffle=True)

        return tra_loader, val_loader, tst_loader, scaler
    def get_raw_data(self, data):
        
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        # add time window
        x_tra, y_tra = Add_Window_Horizon(data_train, window=args.lag, horizon=args.horizon, single=False)
        x_val, y_val = Add_Window_Horizon(data_val, window=args.lag, horizon=args.horizon, single=False)
        x_test, y_test = Add_Window_Horizon(data_test, window=args.lag, horizon=args.horizon, single=False)
        
        print('Train: ', x_tra.shape, y_tra.shape)
        print('Val: ', x_val.shape, y_val.shape)
        print('Test: ', x_test.shape, y_test.shape)
        pro_path = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon))
        if os.path.exists(pro_path) is False:
            os.makedirs(pro_path) 
        tra_path = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon), 'tra_graph_lag{}_hoz{}.mat'.format(args.lag, args.horizon))
        val_path = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon), 'val_graph_lag{}_hoz{}.mat'.format(args.lag, args.horizon))
        tst_path = os.path.join('./data/process', args.dataset, 'lag{}_hoz{}'.format(args.lag, args.horizon), 'tst_graph_lag{}_hoz{}.mat'.format(args.lag, args.horizon))
        io.savemat(tra_path, {'X': x_tra, 'Y': y_tra})
        io.savemat(val_path, {'X': x_val, 'Y': y_val})
        io.savemat(tst_path, {'X': x_test, 'Y': y_test})

    
    def load_st_dataset(self, dataset):
        # output B, N, D
        if dataset == 'PEMS4':
            data_path = os.path.join('./data/PEMS04/PEMS04.npz')
            data = np.load(data_path)['data'][:, :, 0]
        elif dataset == 'PEMS8':
            data_path = os.path.join('./data/PEMS08/PEMS08.npz')
            data = np.load(data_path)['data'][:, :, 0]
        elif dataset == 'PEMS3':
            data_path = os.path.join('../data/PEMS03/PEMS03.npz')
            data = np.load(data_path)['data'][:, :, 0]
        elif dataset == 'PEMS7':
            data_path = os.path.join('../data/PEMS07/PEMS07.npz')
            data = np.load(data_path)['data'][:, :, 0]
        else:
            raise ValueError
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data
    def build_adj(self, ):
        adj_ori = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename)
        pad_adj = np.zeros((int(args.num_nodes), int(args.num_nodes)),
                 dtype=np.float32)
        adj_row = [adj_ori] +  [pad_adj]*(args.lag -1)
        adj_row = np.concatenate(adj_row, axis=1)
        adj = adj_row
        for idx in range(args.lag - 1):
            adj_new = np.roll(adj_row, args.num_nodes*(idx+1), axis=1)
            adj = np.vstack((adj, adj_new))
        return adj            
    
    def eval_adj(self, adj):
        plt.figure(figsize=(5, 5))
        sns.heatmap(adj, cmap=plt.get_cmap('viridis', 6), center=None, robust=False, square=True, xticklabels=False, yticklabels=False)##30
        # plt.title("")
        plt.tight_layout()
        plt.savefig('./fig/{}.png'.format(args.dataset))

    
