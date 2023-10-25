import copy
import argparse
import random
import numpy as np
import os
import pandas as pd
import networkx as nx
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, negative_sampling, subgraph
from torch_geometric.nn import InstanceNorm
from .metaModel import LinearCustom, GINConvCustom, ParameterGenerator
from Params import args


class SequentialCustom(Sequential):
    def forward(self, input, params):
        param_idx = 0
        for module in self:
            if isinstance(module, LinearCustom):
                input = module(input, params[param_idx])
                param_idx += 1
            else:
                input = module(input)
        return input

class GIN_NodeWeightEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, add_mask=False):
        super().__init__()
        # num_features = dataset_num_features

        # nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        nn1 = SequentialCustom(LinearCustom(), ReLU(), LinearCustom())
        self.conv1 = GINConvCustom(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn5 = None
        if add_mask == True:
            # nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 3))
            nn5 = SequentialCustom(LinearCustom(), ReLU(), LinearCustom())
            self.conv5 = GINConvCustom(nn5)
            self.bn5 = torch.nn.BatchNorm1d(3)
        else:
            nn5 = SequentialCustom(LinearCustom(), ReLU(), LinearCustom())
            self.conv5 = GINConvCustom(nn5)
            self.bn5 = torch.nn.BatchNorm1d(2)
    
    def forward(self, x_in, edge_index_in, params_linear):  
        """conv

        Args:
            x_in (_type_): N, feas
            edge_index_in (_type_): 2, edge_num

        Returns:
            _type_: _description_
        """
        x, edge_index = x_in, edge_index_in
        x = F.relu(self.conv1(x, edge_index, nn_params = params_linear[:2]))
        x = self.bn1(x)
        hid_x_rep = x # 
        # TODO: add multi-layers encoder
        x = F.relu(self.conv5(x, edge_index, nn_params = params_linear[2:]))
        x = self.bn5(x)
        return x, hid_x_rep
class BatchSequential(nn.Sequential):
    def forward(self, inputs, params):
        param_idx = 0
        for module in self._modules.values():
            if isinstance(module, (LinearCustom)):
                # print('inputs: {}, batch: {}'.format(inputs.shape, batch.shape))
                inputs = module(inputs, params[param_idx])
                param_idx += 1
            else:
                inputs = module(inputs)
        return inputs

class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(LinearCustom())

            if i < len(channels) - 1:
                # m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        dropout_p = args.extractor_drop

        self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 2], dropout=dropout_p)

    def forward(self, emb, edge_index, params_linear):
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        # print(batch[col])

        # att_log_logits = self.feature_extractor(f12, batch[col])
        att_log_logits = self.feature_extractor(f12, params_linear)
        return att_log_logits

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class ViewGenerator(VGAE):
    def __init__(self, num_features, dim, encoder, edge_extractor,  add_mask=False, num_nodes = None):
        self.add_mask = add_mask
        
        encoder = encoder(num_features, dim, self.add_mask)
        super().__init__(encoder=encoder)
        self.edge_extractor = edge_extractor(dim)
        # -- for gcn
        self.gcn_params_gen = nn.ModuleList()
        self.gcn_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = num_features, output_dim = dim, num_nodes = num_nodes, dynamic = args.dynamic))
        self.gcn_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = dim, output_dim = dim, num_nodes = num_nodes, dynamic = args.dynamic))
        self.gcn_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = dim, output_dim = dim, num_nodes = num_nodes, dynamic = args.dynamic))
        self.gcn_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = dim, output_dim = 3, num_nodes = num_nodes, dynamic = args.dynamic))

        # -- for MLP
        self.mlp_params_gen = nn.ModuleList()
        self.mlp_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = dim*2, output_dim = dim*4, num_nodes = num_nodes, dynamic = args.dynamic))
        self.mlp_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = dim*4, output_dim = dim, num_nodes = num_nodes, dynamic = args.dynamic))
        self.mlp_params_gen.append(ParameterGenerator(memory_size = args.mem_size, input_dim = dim, output_dim = 2, num_nodes = num_nodes, dynamic = args.dynamic))

        # -- for
        if args.dynamic:
            self.mu_estimator = nn.Sequential(*[
                nn.Linear(num_features*num_nodes, 32), # B, N ,32
                nn.Tanh(),
                nn.Linear(32, 32), # B, N, 32
                nn.Tanh(),
                nn.Linear(32, args.mem_size)# B, N, M
            ])

            self.logvar_estimator = nn.Sequential(*[
                nn.Linear(num_features*num_nodes, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, args.mem_size) # B, N, M
            ])
            self.mu = nn.Parameter(torch.randn(1, args.mem_size).to(args.device), requires_grad=True).to(args.device)
            self.logvar = nn.Parameter(torch.randn(1, args.mem_size).to(args.device), requires_grad=True).to(args.device)
    
    def forward(self, x_in, edge_index_in, requires_grad):
        '''
        x_in: B, T, N, 3 or B, N, T, 3
        edge_index_in: 2, T/N
        1) spat-edge aug : 1-hop mask, 2-hop mask, ...
        2) spat-node aug : key, mask, drop
        3) temp-edge aug : mask
        4) temp-node aug : key, mask, drop
        return keep sample
        TODO: drop node?
        '''
        batch_size = x_in.shape[0]
        num_nodes = x_in.shape[1]
        x_in = x_in.reshape(batch_size, num_nodes, -1)

        x = copy.deepcopy(x_in) # B, N, 

        edge_index = copy.deepcopy(edge_index_in) # 2, edge_num
        ##TODO: add multi-hops edges
        # node encoding, and edge encoding
        x_encode = x_in.float() # B, N, F
        edge_idx_encode = edge_index_in

        # -- gen params:
        if args.dynamic:
            mu = self.mu_estimator(x_encode.reshape(batch_size, -1))
            logvar = self.logvar_estimator(x_encode.reshape(batch_size, -1))
            z_data = reparameterize(mu, logvar) # B, M
            z_sample = reparameterize(self.mu, self.logvar)
            z_data = z_data + z_sample # B, M
        else:
            z_data = 0

        gcn_params =[layer(x_encode, z_data) for layer in self.gcn_params_gen]
        # [B*N*in_dim*out_dim, B*N*in_dim*out_dim, B*N*in_dim*out_dim, B*N*in_dim*out_dim]
        mlp_params =[layer(x_encode, z_data) for layer in self.mlp_params_gen]
        # [B*N*in_dim*out_dim, B*N*in_dim*out_dim, B*N*in_dim*out_dim]

        p_node_batch = []
        p_edge_batch = []
        for bch in range(batch_size):
            p_node, hid_x_rep = self.encoder(x_encode[bch], edge_idx_encode, params_linear = [[gcn_params[idx][0][bch], gcn_params[idx][1][bch]] for idx in range(len(gcn_params))]) 
            p_edge = self.edge_extractor(hid_x_rep, edge_index, params_linear = [[mlp_params[idx][0][bch], mlp_params[idx][1][bch]] for idx in range(len(mlp_params))])
            p_node_batch.append(p_node)
            p_edge_batch.append(p_edge)
        p_node_batch = torch.stack(p_node_batch, dim = 0) # B, N, 3
        p_edge_batch = torch.stack(p_edge_batch, dim = 0) # B, E, 2

        x = x.float()
        x.requires_grad = requires_grad
        
        sample_node = F.gumbel_softmax(p_node_batch, hard=True) 
        # B, N, 3
        # 0: reserve, 1: drop, 3: mask
        sample_edge = F.gumbel_softmax(p_edge_batch, hard=True)
        # B, E, 2

        edge_idx_batch = []
        for bch in range(batch_size):
            sample_edge_item = sample_edge[bch, : , 0]
            # E
            edge_idx_tmp = copy.deepcopy(edge_index_in)
            reser_edge_idx = torch.nonzero(sample_edge_item)
            edge_idx_tmp = edge_idx_tmp[:, reser_edge_idx]
            edge_idx_batch.append(copy.deepcopy(edge_idx_tmp))

        keep_sample_batch = []
        mask_sample_batch = []
        for bch in range(batch_size):
            # B, N, 3
            # 0: reserve, 1: drop, 2: mask
            edge_idx_item = edge_idx_batch[bch]
            sample_node_item = sample_node[bch]
            sample_node_reser = sample_node_item[:, 0]
            sample_node_mask = None
            if self.add_mask == True:
                sample_node_mask = sample_node_item[:, 2]
                sample_node_keep = sample_node_mask + sample_node_reser # N
            else:
                sample_node_keep = sample_node_reser # N
            node_keep_idx = torch.nonzero(sample_node_keep, as_tuple=False).view(-1,)
            edge_idx_new, _ = subgraph(node_keep_idx, edge_idx_item, num_nodes=num_nodes)
            edge_idx_batch[bch] = copy.deepcopy(edge_idx_new) # final edge_idx
            keep_sample_batch.append(sample_node_keep)
            mask_sample_batch.append(sample_node_mask)
        
        return edge_idx_batch, keep_sample_batch, mask_sample_batch



def add_distant_neighbors(data, hops):
    """Add multi_edge_index attribute to data which includes the edges of 2,3,... hops neighbors."""
    assert hops > 1
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index,
                                   num_nodes=data.x.size(0))
    one_hop_set = set([tuple(x) for x in edge_index.transpose(0, 1).tolist()])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col)
    multi_adj = adj
    for _ in range(hops - 1):
        multi_adj = multi_adj @ adj
    row, col, _ = multi_adj.coo()
    multi_edge_index = torch.stack([row, col], dim=0)
    multi_hop_set = set([tuple(x) for x in multi_edge_index.transpose(0, 1).tolist()])
    multi_hop_set = multi_hop_set - one_hop_set
    multi_edge_index = torch.LongTensor(list(multi_hop_set)).transpose(0, 1)
    data.multi_edge_index = multi_edge_index
    return