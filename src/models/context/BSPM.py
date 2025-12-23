# -*- coding: UTF-8 -*-

"""
BSPM (Blurring-Sharpening Process Model)
Reference:
    "BSPM: Blurring-Sharpening Process Models for Dynamic Graph Recommendation"
    Original BSPM implementation adapted for ReChorus framework.
CMD example:
    python src/main.py --model_name BSPM --workers 4 --train 0 --dataset Grocery_and_Gourmet_Food
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import time
import torch
import torch.nn as nn
from torchdiffeq import odeint

from models.BaseModel import GeneralModel

class BSPMBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--factor_dim', type=int, default=100,
                            help='Dimension of factor decomposition.')
        parser.add_argument('--idl_beta', type=float, default=0.3,
                            help='Beta parameter for IDL process.')
        parser.add_argument('--T_idl', type=float, default=1.0,
                            help='Time span for IDL process.')
        parser.add_argument('--K_idl', type=int, default=1,
                            help='Number of steps for IDL process.')
        parser.add_argument('--T_b', type=float, default=1.0,
                            help='Time span for blurring process.')
        parser.add_argument('--K_b', type=int, default=1,
                            help='Number of steps for blurring process.')
        parser.add_argument('--T_s', type=float, default=1.0,
                            help='Time span for sharpening process.')
        parser.add_argument('--K_s', type=int, default=1,
                            help='Number of steps for sharpening process.')
        parser.add_argument('--solver_idl', type=str, default='rk4',
                            help='Solver for IDL ODE.')
        parser.add_argument('--solver_blr', type=str, default='rk4',
                            help='Solver for blurring ODE.')
        parser.add_argument('--solver_shr', type=str, default='rk4',
                            help='Solver for sharpening ODE.')
        parser.add_argument('--final_sharpening', type=int, default=1,
                            help='Whether to use final sharpening.')
        parser.add_argument('--sharpening_off', type=int, default=0,
                            help='Whether to turn off sharpening.')
        parser.add_argument('--t_point_combination', type=int, default=0,
                            help='Whether to combine time points.')
        return parser

    def _base_init(self, args, corpus):
        self.factor_dim = args.factor_dim
        self.idl_beta = args.idl_beta
        self.T_idl = args.T_idl
        self.K_idl = args.K_idl
        self.T_b = args.T_b
        self.K_b = args.K_b
        self.T_s = args.T_s
        self.K_s = args.K_s
        self.solver_idl = args.solver_idl
        self.solver_blr = args.solver_blr
        self.solver_shr = args.solver_shr
        self.final_sharpening = args.final_sharpening
        self.sharpening_off = args.sharpening_off
        self.t_point_combination = args.t_point_combination
        self.item_num = corpus.n_items
        
        # Initialize time points for ODE solvers
        self.idl_times = torch.linspace(0, self.T_idl, self.K_idl + 1).float()
        self.blurring_times = torch.linspace(0, self.T_b, self.K_b + 1).float()
        self.sharpening_times = torch.linspace(0, self.T_s, self.K_s + 1).float()
        
        
        print(f"BSPM Configuration:")
        print(f"  IDL: {self.solver_idl}, BLR: {self.solver_blr}, SHR: {self.solver_shr}")
        print(f"  IDL factor_dim: {self.factor_dim}")
        print(f"  IDL beta: {self.idl_beta}")
        print(f"  IDL time: {self.idl_times}")
        print(f"  Blur time: {self.blurring_times}")
        print(f"  Sharpen time: {self.sharpening_times}")
        print(f"  Final sharpening: {self.final_sharpening}")
        print(f"  Sharpening off: {self.sharpening_off}")
        print(f"  Time point combination: {self.t_point_combination}")
        
        # Build adjacency matrix and preprocess
        self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        # self.norm_adj_sparse = self._scipy_to_torch_sparse(self.norm_adj)

    def _scipy_to_torch_sparse(self, scipy_mat):
        """Convert a scipy sparse matrix to a torch sparse tensor on demand.

        IMPORTANT: do not store torch.sparse tensors as model attributes because
        they are not picklable and cause multiprocessing (spawn) to fail on Windows.
        Instead cache COO numpy arrays (which are picklable) and build a torch
        sparse tensor each time this function is called.
        """
        if not hasattr(self, '_cached_norm_adj_coo'):
            if scipy_mat.format != 'coo':
                scipy_mat = scipy_mat.tocoo()
            # store as numpy arrays (pickle-friendly)
            self._cached_norm_adj_coo = (
                scipy_mat.row.astype(np.int64),
                scipy_mat.col.astype(np.int64),
                scipy_mat.data.astype(np.float32),
                tuple(scipy_mat.shape)
            )

        rows, cols, data, shape = self._cached_norm_adj_coo
        indices = torch.from_numpy(np.vstack((rows, cols))).long()
        values = torch.from_numpy(data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    
    def build_adjmat(self, user_count, item_count, train_mat):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        adj_mat = R.tocsr()
        self.adj_mat = adj_mat

        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1)) + 1e-10
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0)) + 1e-10
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = svds(self.norm_adj, k=self.factor_dim)
        end = time.time()
        print('training time for BSPM', end-start)
    
    def IDLFunction2(self, t, r):
        """IDL (Item Diffusion Learning) ODE function."""
        # Convert sparse matrices to dense tensors for PyTorch operations
        d_mat_i_dense = torch.FloatTensor(self.d_mat_i.toarray())
        d_mat_i_inv_dense = torch.FloatTensor(self.d_mat_i_inv.toarray())
        vt_tensor = torch.FloatTensor(self.vt)
        
        out = r @ d_mat_i_dense @ vt_tensor.T @ vt_tensor @ d_mat_i_inv_dense
        out = out - r
        return out
    
    def IDLFunction(self, t, r):
        """IDL (Item Diffusion Learning) ODE function."""
        # 使用对角线元素代替完整矩阵
        d_i_diag = torch.FloatTensor(self.d_mat_i.diagonal())
        d_i_inv_diag = torch.FloatTensor(self.d_mat_i_inv.diagonal())
        vt_tensor = torch.FloatTensor(self.vt)  # shape: (factor_dim, item_num)

        # 分步计算避免大矩阵
        step1 = r * d_i_diag  # 等效于 r @ d_mat_i_dense
        step2 = step1 @ vt_tensor.T  # (batch, factor_dim)
        step3 = step2 @ vt_tensor  # (batch, item_num)
        step4 = step3 * d_i_inv_diag  # 等效于 step3 @ d_mat_i_inv_dense

        out = step4 - r
        return out

    
    def blurFunction2(self, t, r):
        """Blurring ODE function."""
        R_dense = torch.FloatTensor(self.norm_adj.toarray())
        out = r @ R_dense.T @ R_dense
        out = out - r
        return out
    
    def sharpenFunction2(self, t, r):
        """Sharpening ODE function."""
        R_dense = torch.FloatTensor(self.norm_adj.toarray())
        out = r @ R_dense.T @ R_dense
        return -out
    
    def blurFunction(self, t, r):
        """Blurring ODE function (最终修正版)."""
        # 每次调用时从可序列化的scipy矩阵创建PyTorch稀疏张量
        # 获取稀疏 COO 张量并确保在 r 相同的设备上
        norm_adj_sparse = self._scipy_to_torch_sparse(self.norm_adj).coalesce().to(r.device)

        # 计算 temp_transpose = norm_adj_sparse @ r.T  -> (user, batch)
        temp_transpose = torch.sparse.mm(norm_adj_sparse, r.T)
        temp = temp_transpose.T  # temp 是 (batch, user)

        idx = norm_adj_sparse.indices()
        vals = norm_adj_sparse.values()
        trans_idx = torch.stack([idx[1, :], idx[0, :]], dim=0)
        trans_shape = (norm_adj_sparse.shape[1], norm_adj_sparse.shape[0])
        norm_adj_sparse_T = torch.sparse_coo_tensor(trans_idx, vals, trans_shape, device=norm_adj_sparse.device).coalesce()

        # out_transpose = norm_adj_sparse_T @ temp.T -> (item, batch)
        out_transpose = torch.sparse.mm(norm_adj_sparse_T, temp.T)
        out = out_transpose.T  # out 是 (batch, item)

        return out - r

    def sharpenFunction(self, t, r):
        """Sharpening ODE function (最终修正版)."""
        norm_adj_sparse = self._scipy_to_torch_sparse(self.norm_adj).coalesce().to(r.device)

        # 与 blurFunction 计算方式相同
        temp_transpose = torch.sparse.mm(norm_adj_sparse, r.T)
        temp = temp_transpose.T

        idx = norm_adj_sparse.indices()
        vals = norm_adj_sparse.values()
        trans_idx = torch.stack([idx[1, :], idx[0, :]], dim=0)
        trans_shape = (norm_adj_sparse.shape[1], norm_adj_sparse.shape[0])
        norm_adj_sparse_T = torch.sparse_coo_tensor(trans_idx, vals, trans_shape, device=norm_adj_sparse.device).coalesce()

        out_transpose = torch.sparse.mm(norm_adj_sparse_T, temp.T)
        out = out_transpose.T

        return -out
    
    def get_user_ratings(self, batch_test):
        # adj_mat = self.adj_mat
        
        with torch.no_grad():
            idl_out = odeint(func=self.IDLFunction, y0=torch.Tensor(batch_test), 
                                t=self.idl_times, method=self.solver_idl)
                
            blurred_out = odeint(func=self.blurFunction, y0=torch.Tensor(batch_test), 
                               t=self.blurring_times, method=self.solver_blr)
            
            # Sharpening process
            if not self.sharpening_off:
                if self.final_sharpening:
                    sharpened_out = odeint(func=self.sharpenFunction, 
                                             y0=self.idl_beta * idl_out[-1] + blurred_out[-1], 
                                             t=self.sharpening_times, method=self.solver_shr)
                else:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out[-1], 
                                         t=self.sharpening_times, method=self.solver_shr)
        
        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 =  torch.mean(torch.cat([blurred_out[1:,...],sharpened_out[1:,...]],axis=0),axis=0)
            else:
                U_2 =  torch.mean(blurred_out[1:,...],axis=0)
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
            else:
                U_2 = blurred_out[-1]
        
        if self.final_sharpening == True:
            if self.sharpening_off == False:
                ret = U_2.numpy()
            elif self.sharpening_off == True:
                ret = U_2.numpy() + self.idl_beta * idl_out[-1].numpy()
        else:
            ret = U_2.numpy() + self.idl_beta * idl_out[-1].numpy()
        return ret


class BSPM(GeneralModel, BSPMBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['factor_dim', 'idl_beta', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser = BSPMBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        """Forward pass for BSPM model."""
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        batch = np.zeros((len(u_ids), self.item_num))
        for i in range(len(u_ids)):
            batch[i,i_ids[i].cpu().data.numpy()] = 1 

        adj_mat = self.adj_mat
        batch = adj_mat[u_ids.cpu().data.numpy(), :].toarray()
        
        # Get user ratings for all items
        user_ratings = self.get_user_ratings(batch)
        user_ratings = torch.FloatTensor(user_ratings).to(self.device)
        
        # Extract predictions for the specific items
        batch_size = len(u_ids)
        n_candidates = i_ids.shape[1]
        
        # Create index arrays for advanced indexing
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_candidates).long()
        candidate_indices = i_ids.long()
        
        # Gather predictions using advanced indexing
        prediction = user_ratings[batch_indices, candidate_indices]
        
        return {'prediction': prediction}

    def predict(self, feed_dict):
        """Prediction method for evaluation."""
        return self.forward(feed_dict)
