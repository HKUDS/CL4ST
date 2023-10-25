import torch
from torch import nn
import time
import numpy as np
import os
from componenets.metrics import metrics
from componenets.new_metrics import metric_new
from methods.cl4st.stgat.stgat import STGAT_cl
from methods.cl4st.view_generator import ViewGenerator, GIN_NodeWeightEncoder, ExtractorMLP
from torch_geometric.utils import dense_to_sparse
import utils.util as util
import pandas as pd
import torch_geometric
import math
from tqdm import tqdm
from scipy.sparse import coo_matrix
import torch.nn.functional as F
from Params import args, logger
def load_SE(num_node, d_model):
    # SE = torch.zeros([num_node, num_node])
    # for ind in num_node:
    #     SE[ind, ind] = 1
    # return SE
    pe = torch.zeros(num_node, d_model)
    position = torch.arange(0, num_node, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

def Beta_Function(x, alpha, beta):
    """Beta function"""
    from scipy.special import gamma
    return gamma(alpha + beta) / gamma(alpha) / gamma(beta) * x ** (alpha - 1) * (1 - x) ** (beta - 1)

def record_metric(data_record_dict, data_list, key_list):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        data_record_dict[key] = data
    return data_record_dict
def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)
def build_sp_tensor(adj_weight):
    coo_adj = coo_matrix(adj_weight)
    sp_adj = convert_sp_mat_to_sp_tensor(coo_adj)
    return sp_adj

def MAE_torch(pred, true, mask_value=0):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))
class trainer():
    def __init__(self, scaler, sp_adj = None, sp_adj_w = None, temp_adj = None):
        self.scaler = scaler
        self.sp_adj = sp_adj
        self.sp_adj_w = sp_adj_w
        self.temp_adj = temp_adj
        SE = load_SE(args.num_nodes, 64)
        SE = SE.to("cuda:0")
        if args.model == 'STGAT':
            self.model = STGAT(sp_adj, sp_adj_w, temp_adj)
        elif args.model == 'cl4st': 
            self.model = STGAT_cl()
            self.init_st_graph(sp_adj, sp_adj_w, temp_adj)
            self.get_view()
            self.get_view_optim()

        else:
            raise ValueError('Model :{} error'.format(args.model))
        
        if args.testonly:
            # self.model.load("checkpoints/TaxiBJ/model_finetune.pth")
            self.model.load(args.mdir+args.name+'.pkl')
            self.model = self.model.to(args.device)
        else:
            self.model = self.model.to(args.device)
        self.optimizer, self.lr_scheduler = self.get_optim()
        self.criterion = self.get_criterion()
        
        # early stop
        self.patience = args.patience 
        self.trigger = 0
        self.last_loss = 100000
        self.last_mape_loss = 100000
        self.best_epoch = 0
        self.best_state = self.model.state_dict()
        self.build_beta_list(args.beta1, args.beta2)
        self.build_coeff_cl_list()
    def init_st_graph(self, sp_adj, sp_adj_w, tem_adj):
        self.edge_idx_spat, self.edge_wg_spat = dense_to_sparse(torch.from_numpy(sp_adj_w))
        tem_adj = np.ones((args.lag, args.lag))
        self.edge_idx_temp, self.edge_wg_temp = dense_to_sparse(torch.from_numpy(tem_adj))
        self.edge_idx_spat = self.edge_idx_spat.to(args.device)
        self.edge_wg_spat = self.edge_wg_spat.to(args.device)
        self.edge_idx_temp = self.edge_idx_temp.to(args.device)
        self.edge_wg_temp = self.edge_wg_temp.to(args.device)
    def get_view(self, ):
        self.view_gen1_spat = ViewGenerator(num_features=args.lag*3, dim = args.view_dim_spat, encoder = GIN_NodeWeightEncoder, edge_extractor = ExtractorMLP,  add_mask=args.add_mask, num_nodes=args.num_nodes)
        self.view_gen1_spat = self.view_gen1_spat.to(args.device)
        if args.view_num == 2:
            self.view_gen2_spat = ViewGenerator(num_features=args.lag*3, dim = args.view_dim_spat, encoder = GIN_NodeWeightEncoder, edge_extractor = ExtractorMLP,  add_mask=args.add_mask, num_nodes=args.num_nodes)
            self.view_gen2_spat = self.view_gen2_spat.to(args.device)
        self.view_gen1_temp = ViewGenerator(num_features=args.num_nodes*3, dim = args.view_dim_temp, encoder = GIN_NodeWeightEncoder, edge_extractor = ExtractorMLP,  add_mask=args.add_mask, num_nodes=args.lag)
        self.view_gen1_temp = self.view_gen1_temp.to(args.device)
        if args.view_num == 2:
            self.view_gen2_temp = ViewGenerator(num_features=args.num_nodes*3, dim = args.view_dim_temp, encoder = GIN_NodeWeightEncoder, edge_extractor = ExtractorMLP,  add_mask=args.add_mask, num_nodes=args.lag)
            self.view_gen2_temp = self.view_gen2_temp.to(args.device)
    def get_view_optim(self, ):
        if args.view_num == 1:
            self.view_optimizer = torch.optim.Adam([ {'params': self.view_gen1_spat.parameters()}, {'params': self.view_gen1_temp.parameters()}], lr=args.lr
                                , weight_decay=args.weight_decay)
        elif args.view_num == 2:
            self.view_optimizer = torch.optim.Adam([ {'params': self.view_gen1_spat.parameters()}, {'params': self.view_gen2_spat.parameters()}, {'params': self.view_gen1_temp.parameters()}, {'params': self.view_gen2_temp.parameters()}], lr=args.lr
                                , weight_decay=args.weight_decay)
        else:
            raise ValueError('view_num must be 1 or 2, can not be {}'.format(args.view_num))
        lr_decay_ratio = args.lr_decay_ratio
        self.lr_scheduler_view = torch.optim.lr_scheduler.MultiStepLR(self.view_optimizer, milestones=args.steps,
                                                                gamma=lr_decay_ratio)


    def decorate_batch(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(args.device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(args.device)
                elif isinstance(value, dict) or isinstance(value, list):
                    batch[key] = self.decorate_batch(value)
                # retain other value types in the batch dict
            return batch
        elif isinstance(batch, list):
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(args.device))
                elif isinstance(value, dict) or isinstance(value, list):
                    new_batch.append(self.decorate_batch(value))
                else:
                    # retain other value types in the batch list
                    new_batch.append(value)
            return new_batch
        elif isinstance(batch, torch_geometric.data.batch.DataBatch):
            return batch.to(args.device)
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))
    
    def cal_batch_sim_loss(self, keep_sample_1, keep_sample_2):
        loss_list = []
        for sample_1, sample_2 in zip(keep_sample_1, keep_sample_2):
            loss_list.append(F.mse_loss(sample_1, sample_2))
        return torch.mean(torch.stack(loss_list, dim=0))
    def cal_batch_cl_loss(self, x1, x2 ):
        # x1, x2 : B, F
        T = args.cl_temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_a = torch.exp(sim_matrix_a / T)
        pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
        loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
        loss_a = - torch.log(loss_a).mean()

        sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
        sim_matrix_b = torch.exp(sim_matrix_b / T)
        pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
        loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
        loss_b = - torch.log(loss_b).mean()

        loss = (loss_a + loss_b) / 2
        return loss
    def org_pre_loss(self, pre_loss_list):
        if args.loss_mode == 'mean_loss':
            pre_loss = 0
            for loss in pre_loss_list:
                pre_loss += loss
            pre_loss = pre_loss /len(pre_loss_list)
        elif args.loss_mode == 'sum_loss':
            pre_loss = 0
            for loss in pre_loss_list:
                pre_loss += loss
        elif args.loss_mode == 'single_loss':
            pre_loss = pre_loss_list[0]
        return pre_loss

    def train(self, epoch, trnloader, tra_val_metric):
        tra_loss = []
        pre_loss = []

        # cl4st:
        pre_loss_1 = []
        pre_loss_2 = []
        sim_loss_spat_list = []
        sim_loss_temp_list = []
        cl_loss_list = []
        KLD_spat_list = []
        KLD_temp_list = []

        self.model.train()
        total_days = (args.end_date - args.start_date).days+1
        ids = np.random.permutation(list(range(args.lag, total_days)))
        num = len(ids)
        beta1 = self.beta1_list[epoch] if self.beta1_list is not None else None
        beta2 = self.beta2_list[epoch] if self.beta1_list is not None else None

        sim_spat_coeff = self.sim_spat_coeff_list[epoch]
        sim_temp_coeff = self.sim_temp_coeff_list[epoch]
        cr = self.cr_list[epoch]

        kld_spat_coeff = self.kld_spat_coeff_list[epoch]
        kld_temp_coeff = self.kld_temp_coeff_list[epoch]

        
        for idx, batch in tqdm(enumerate(trnloader)):
            # reg_info = dict()
            self.view_optimizer.zero_grad()
            self.optimizer.zero_grad()
            batch = self.decorate_batch(batch)
            X, Y, TE = batch
            # print('feats :{}'.format(feats.shape)) # 58944, 1
            # print('edge_index :{}'.format(bg.edge_index.shape)) # 2, 130560
            # print(X)
            if args.model == 'STGAT':
                t_emb = TE[:, :args.lag, :] # B, T, 2
                t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                X = torch.cat([X, t_emb], dim = -1)
                output = self.model(X, epoch=epoch)
            elif args.model == 'cl4st':
                t_emb = TE[:, :args.lag, :] # B, T, 2
                t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                X = torch.cat([X, t_emb], dim = -1) # B, T, N, 3
                # -- spatial view
                X_spat = X.transpose(2, 1)
                batch_size = X.shape[0]
                edge_idx_batch_spat_1, keep_sample_batch_spat_1, mask_sample_batch_spat_1 = self.view_gen1_spat(x_in = X_spat, edge_index_in = self.edge_idx_spat, requires_grad = True)

                if args.view_num == 2:
                    edge_idx_batch_spat_2, keep_sample_batch_spat_2, mask_sample_batch_spat_2 = self.view_gen2_spat(x_in = X_spat, edge_index_in = self.edge_idx_spat, requires_grad = True)

                # -- temporal view
                X_temp = X
                edge_idx_batch_temp_1, keep_sample_batch_temp_1, mask_sample_batch_temp_1 = self.view_gen1_temp(x_in = X_temp, edge_index_in = self.edge_idx_temp, requires_grad = True)
                if args.view_num == 2:
                    edge_idx_batch_temp_2, keep_sample_batch_temp_2, mask_sample_batch_temp_2 = self.view_gen2_temp(x_in = X_temp, edge_index_in = self.edge_idx_temp, requires_grad = True)
                
                edge_idx_batch_spat_ori = [self.edge_idx_spat]*batch_size
                edge_idx_batch_temp_ori = [self.edge_idx_temp]*batch_size

                X.requires_grad = True
                output_ori, output_rep_ori = self.model(X, epoch=epoch, edge_idx_spat = edge_idx_batch_spat_ori, edge_idx_temp = edge_idx_batch_temp_ori)

                output_1, output_rep_1 = self.model(X, epoch=epoch, edge_idx_spat = edge_idx_batch_spat_1, edge_idx_temp = edge_idx_batch_temp_1, keep_sample_spat = keep_sample_batch_spat_1, keep_sample_temp = keep_sample_batch_temp_1, mask_sample_spat = mask_sample_batch_spat_1, mask_sample_temp = mask_sample_batch_temp_1)
                if args.view_num == 2:
                    output_2, output_rep_2 = self.model(X, epoch=epoch, edge_idx_spat = edge_idx_batch_spat_2, edge_idx_temp = edge_idx_batch_temp_2, keep_sample_spat = keep_sample_batch_spat_2, keep_sample_temp = keep_sample_batch_temp_2, mask_sample_spat = mask_sample_batch_spat_2, mask_sample_temp = mask_sample_batch_temp_2)

            
            if args.model == 'STGAT':
                output  = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                main_loss = self.criterion(output, Y)
                loss = main_loss
                pre_loss.append(main_loss.item())
            elif args.model == 'cl4st':
                output_ori  = self.scaler.inverse_transform(output_ori)
                output_1  = self.scaler.inverse_transform(output_1)
                if args.view_num == 2:
                    output_2  = self.scaler.inverse_transform(output_2)
                Y = self.scaler.inverse_transform(Y)
                ## pre loss
                main_loss_ori = self.criterion(output_ori, Y)
                main_loss_1 = self.criterion(output_1, Y)
                if args.view_num == 2:
                    main_loss_2 = self.criterion(output_2, Y)
                else:
                    main_loss_2 = torch.Tensor([0])
                ## sim loss
                # -- spat
                if args.view_num == 2:
                    sim_loss_spat = self.cal_batch_sim_loss(keep_sample_batch_spat_1, keep_sample_batch_spat_2)
                    sim_loss_spat = (1 - sim_loss_spat)
                else:
                    sim_loss_spat = 0
                # -- temp
                if args.view_num == 2:
                    sim_loss_temp = self.cal_batch_sim_loss(keep_sample_batch_temp_1, keep_sample_batch_temp_2)
                    sim_loss_temp = (1 - sim_loss_temp)
                else:
                    sim_loss_temp = 0
                ## cl loss 
                if args.view_num == 2:
                    cl_loss = self.cal_batch_cl_loss(output_rep_1, output_rep_2)
                elif args.view_num == 1:
                    cl_loss = self.cal_batch_cl_loss(output_rep_ori, output_rep_1)

                if args.view_num == 2:
                    main_loss = self.org_pre_loss([main_loss_ori, main_loss_1, main_loss_2])
                else: 
                    main_loss = self.org_pre_loss([main_loss_ori, main_loss_1])

                # -- KL loss for param gen
                view_gen = []
                view_gen.append(self.view_gen1_spat)
                if args.view_num == 2:
                    view_gen.append(self.view_gen2_spat)
                view_gen.append(self.view_gen1_temp)
                if args.view_num == 2:
                    view_gen.append(self.view_gen2_temp)
                if args.dynamic:
                    KLD_spat = []
                    for gen in view_gen[:int(len(view_gen)/2)]:
                        logvar = gen.logvar
                        mu = gen.mu
                        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        mu = gen.mu_estimator(X_spat.reshape(batch_size, -1))
                        logvar = gen.logvar_estimator(X_spat.reshape(batch_size, -1))
                        data_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        KLD = KLD + data_KLD
                        KLD_spat.append(KLD)
                    KLD_temp = []
                    for gen in view_gen[int(len(view_gen)/2):]:
                        logvar = gen.logvar
                        mu = gen.mu
                        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        mu = gen.mu_estimator(X_temp.reshape(batch_size, -1))
                        logvar = gen.logvar_estimator(X_temp.reshape(batch_size, -1))
                        data_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        KLD = KLD + data_KLD
                        KLD_temp.append(KLD)
                    KLD_spat = torch.mean(torch.stack(KLD_spat))
                    KLD_temp = torch.mean(torch.stack(KLD_temp))
                else:
                    KLD_spat = torch.Tensor([0.]).to(args.device)
                    KLD_temp = torch.Tensor([0.]).to(args.device)

                ## total

                loss = main_loss + sim_loss_spat * sim_spat_coeff + sim_loss_temp * sim_temp_coeff + cl_loss * cr + KLD_spat * kld_spat_coeff + KLD_temp * kld_temp_coeff
                pre_loss.append(main_loss_ori.item())
                pre_loss_1.append(main_loss_1.item())
                pre_loss_2.append(main_loss_2.item())
                if args.view_num == 2:
                    sim_loss_spat_list.append(sim_loss_spat.item())
                    sim_loss_temp_list.append(sim_loss_temp.item())
                else:
                    sim_loss_spat_list.append(0)
                    sim_loss_temp_list.append(0)
                cl_loss_list.append((cl_loss * cr).item())
                KLD_spat_list.append((KLD_spat * kld_spat_coeff).item())
                KLD_temp_list.append((KLD_temp * kld_temp_coeff).item())
            
            loss.backward()
            
            # add max grad clipping
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.view_optimizer.step()
            self.optimizer.step()
            tra_loss.append(loss.item())
            pre_lr = self.optimizer.param_groups[0]['lr']
            view_lr = self.view_optimizer.param_groups[0]['lr']
        self.lr_scheduler_view.step()
        self.lr_scheduler.step()
        tra_loss = np.mean(tra_loss)
        pre_loss = np.mean(pre_loss)
        # other_loss = np.mean(other_loss)
        # gsat_loss_list = np.mean(gsat_loss_list)
        if args.model == 'STGAT':
            
            tra_val_metric = record_metric(tra_val_metric, [epoch, tra_loss, pre_loss], ['epoch', 'train loss', 'predict loss'])
        elif args.model == 'cl4st':
            pre_loss_1 = np.mean(pre_loss_1)
            pre_loss_2 = np.mean(pre_loss_2)
            sim_loss_spat_list = np.mean(sim_loss_spat_list)
            sim_loss_temp_list = np.mean(sim_loss_temp_list)
            cl_loss_list = np.mean(cl_loss_list)
            KLD_spat_list = np.mean(KLD_spat_list)
            KLD_temp_list = np.mean(KLD_temp_list)
            tra_val_metric = record_metric(tra_val_metric, [epoch, tra_loss, pre_loss, pre_loss_1, pre_loss_2, sim_loss_spat_list, sim_loss_temp_list, cl_loss_list, pre_lr, view_lr, KLD_spat_list, KLD_temp_list], ['epoch', 'train loss', 'predict loss', 'predict loss 1', 'predict loss 2', 'sim loss spat', 'sim loss temp', 'cl loss', 'pre lr', 'view_lr', 'KLD_spat', 'KLD_temp'])
        else: 
            raise ValueError('Model :{} error, in Display Loss'.format(args.model))
        return tra_val_metric

    def build_beta_list(self, beta1=0.001, beta2 = 0.01, ):
        beta_init = 0
        init_length = int(args.max_epoch / 4)
        anneal_length = int(args.max_epoch / 4)
        beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
        beta1_inter = beta_inter / 4 * (beta_init - beta1) + beta1
        self.beta1_list = np.concatenate([np.ones(init_length) * beta_init, beta1_inter, 
                                     np.ones(args.max_epoch - init_length - anneal_length + 1) * beta1])
                        
        beta_init = 0
        init_length = int(args.max_epoch / 4)
        anneal_length = int(args.max_epoch / 4)
        beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
        beta2_inter = beta_inter / 4 * (beta_init - beta2) + beta2
        self.beta2_list = np.concatenate([np.ones(init_length) * beta_init, beta2_inter, 
                                     np.ones(args.max_epoch - init_length - anneal_length + 1) * beta2])

    def build_coeff_cl_list(self, ):
        sim_spat_coeff_init = args.sim_spat_coeff_init
        if args.is_anneal:
            init_length = int(args.max_epoch / 4)
            anneal_length = int(args.max_epoch / 4)
            beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
            sim_spat_coeff_inter = beta_inter / 4 * (sim_spat_coeff_init - args.sim_spat_coeff) + args.sim_spat_coeff
            self.sim_spat_coeff_list = np.concatenate([np.ones(init_length) * sim_spat_coeff_init, sim_spat_coeff_inter, np.ones(args.max_epoch - init_length - anneal_length + 1) * args.sim_spat_coeff])
        else:
            self.sim_spat_coeff_list = np.ones(args.max_epoch + 1) * args.sim_spat_coeff
                        
        sim_temp_coeff_init = args.sim_temp_coeff_init
        if args.is_anneal:
            init_length = int(args.max_epoch / 4)
            anneal_length = int(args.max_epoch / 4)
            beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
            sim_temp_coeff_inter = beta_inter / 4 * (sim_temp_coeff_init - args.sim_temp_coeff) + args.sim_temp_coeff
            self.sim_temp_coeff_list = np.concatenate([np.ones(init_length) * sim_temp_coeff_init, sim_temp_coeff_inter, np.ones(args.max_epoch - init_length - anneal_length + 1) * args.sim_temp_coeff])
        else:
            self.sim_temp_coeff_list = np.ones(args.max_epoch + 1) * args.sim_temp_coeff

        cr_init = args.cr_init
        if args.is_anneal:
            init_length = int(args.max_epoch / 4)
            anneal_length = int(args.max_epoch / 4)
            beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
            cr_inter = beta_inter / 4 * (cr_init - args.cr) + args.cr
            self.cr_list = np.concatenate([np.ones(init_length) * cr_init, cr_inter, np.ones(args.max_epoch - init_length - anneal_length + 1) * args.cr])
        else:
            self.cr_list = np.ones(args.max_epoch + 1) * args.cr

        kld_spat_coeff_init = args.kld_spat_coeff_init
        if args.is_anneal_kld:
            init_length = int(args.max_epoch / 4)
            anneal_length = int(args.max_epoch / 4)
            beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
            kld_spat_coeff_inter = beta_inter / 4 * (kld_spat_coeff_init - args.kld_spat_coeff) + args.kld_spat_coeff
            self.kld_spat_coeff_list = np.concatenate([np.ones(init_length) * kld_spat_coeff_init, kld_spat_coeff_inter, np.ones(args.max_epoch - init_length - anneal_length + 1) * args.kld_spat_coeff])
        else:
            self.kld_spat_coeff_list = np.ones(args.max_epoch + 1) * args.kld_spat_coeff
                        
        kld_temp_coeff_init = args.kld_temp_coeff_init
        if args.is_anneal_kld:
            init_length = int(args.max_epoch / 4)
            anneal_length = int(args.max_epoch / 4)
            beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
            kld_temp_coeff_inter = beta_inter / 4 * (kld_temp_coeff_init - args.kld_temp_coeff) + args.kld_temp_coeff
            self.kld_temp_coeff_list = np.concatenate([np.ones(init_length) * kld_temp_coeff_init, kld_temp_coeff_inter, np.ones(args.max_epoch - init_length - anneal_length + 1) * args.kld_temp_coeff])
        else:
            self.kld_temp_coeff_list = np.ones(args.max_epoch + 1) * args.kld_temp_coeff


    
    def validation(self,  epoch, valloader, tra_val_metric):
        val_loss = []
        trues = []
        preds = []
        ids = np.random.permutation(list(range(args.lag, 921)))
        num = len(ids)
        with torch.no_grad():
            self.model.eval()

            for idx, batch in tqdm(enumerate(valloader)):
                batch = self.decorate_batch(batch)
                X, Y, TE = batch
                # print('feats :{}'.format(feats.shape))
                if args.model == 'STGAT':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X, epoch=epoch)
                elif args.model == 'cl4st':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    batch_size = X.shape[0]
                    edge_idx_batch_spat_ori = [self.edge_idx_spat]*batch_size
                    edge_idx_batch_temp_ori = [self.edge_idx_temp]*batch_size
                    output, output_rep = self.model(X, epoch=epoch, edge_idx_spat = edge_idx_batch_spat_ori, edge_idx_temp = edge_idx_batch_temp_ori)

                output = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                if args.model != 'STHSL':
                    loss = self.criterion(output, Y)
                # loss = self.criterion(output, Y)
                val_loss.append(loss.item())
                
                trues.append(Y.detach().cpu().numpy())
                preds.append(output.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        trues, preds = np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)
        print(trues.shape, preds.shape)
        mae, rmse, mape, smape, corr = metrics(preds, trues, args.mae_thresh, args.mape_thresh)
        tra_val_metric = record_metric(tra_val_metric, [val_loss, mae, rmse, mape*100, smape*100, corr], ['val loss', 'mae', 'rmse', 'mape(%)', 'smape(%)', 'corr'])
        
        # stopFlg = self.earlyStop( epoch, mae, mape)
        stopFlg = self.earlyStop( epoch, mape, mape)

        return tra_val_metric, stopFlg

    def test(self, tstloader, ):
        self.model.load_state_dict(torch.load(args.mdir+args.name+'.pkl'), False)

        trues = []
        preds = []
        # trues_torch = []
        # preds_torch = []
        ids = np.random.permutation(list(range(args.lag, 921)))
        num = len(ids)

        with torch.no_grad():
            self.model.eval()

            for idx, batch in enumerate(tstloader):
                batch = self.decorate_batch(batch)
                X, Y, TE = batch
                if args.model == 'STGAT':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X, epoch=0)
                elif args.model == 'cl4st':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    batch_size = X.shape[0]
                    edge_idx_batch_spat_ori = [self.edge_idx_spat]*batch_size
                    edge_idx_batch_temp_ori = [self.edge_idx_temp]*batch_size
                    output, output_rep = self.model(X, epoch=0, edge_idx_spat = edge_idx_batch_spat_ori, edge_idx_temp = edge_idx_batch_temp_ori)

                output = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                
                trues.append(Y.detach().cpu().numpy())
                preds.append(output.detach().cpu().numpy())
                # trues_torch.append(Y)
                # preds_torch.append(output)

        # val_loss = np.mean(val_loss)
        trues, preds = np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)
        for t in range(trues.shape[1]):
            mae, rmse, mape, smape, corr = metrics(preds[:, t, ...], trues[:, t, ...], args.mae_thresh, args.mape_thresh)
            log = "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
                t + 1, mae, rmse, mape * 100, smape * 100, corr)
            logger.info(log)
        mae, rmse, mape, smape, corr = metrics(preds, trues, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, Best Epoch: {}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
            self.best_epoch, mae, rmse, mape * 100, smape * 100, corr))
        # preds_tch, trues_ch = torch.cat(preds_torch, dim = 0), torch.cat(trues_torch, dim = 0)
        # mae, mape, rmse= metric_new(preds_tch, trues_ch)
        # logger.info("Average Horizon, New Metrics: {}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        #     self.best_epoch, mae, rmse, mape))




    def get_optim(self, ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay = args.weight_decay, betas=(0.9, 0.999))
        steps = args.steps
        
        lr_decay_ratio = args.lr_decay_ratio
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                                gamma=lr_decay_ratio)
        return optimizer, lr_scheduler

    def get_criterion(self, ):
        if args.criterion == 'MSE':
            return nn.MSELoss()
        elif args.criterion == 'Smooth':
            return nn.SmoothL1Loss()
        elif args.criterion == 'MAE':
            return MAE_torch
    

    def earlyStop(self, epoch, current_loss, mape_loss):
        if epoch >= 100:
            if current_loss >= self.last_loss or epoch == args.max_epoch:
        # if epoch >= 0:
        #     if epoch == 1:
                if current_loss < self.last_loss:
                    self.trigger = 0
                    self.last_loss = current_loss
                    self.last_mape_loss = mape_loss
                    self.best_epoch = epoch
                    self.best_state = self.model.state_dict()
                else:
                    self.trigger += 1
                if self.trigger >= self.patience or epoch == args.max_epoch:   
                    print('Early Stopping! The best epoch is ' + str(self.best_epoch))
                    if not os.path.exists(args.mdir):
                        os.makedirs(args.mdir)
                    torch.save(self.best_state,args.mdir+args.name+'.pkl')
                    return True
            else:
                self.trigger = 0
                self.last_loss = current_loss
                self.last_mape_loss = mape_loss
                self.best_epoch = epoch
                self.best_state = self.model.state_dict()
                return False