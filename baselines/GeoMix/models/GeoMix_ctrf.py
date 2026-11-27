import math
import time
import torch
from torch import nn
from torch_scatter import scatter_mean

from models.modules.EGNN import EGNN
from models.modules.TFN import TFN
from models.utils.mlp import BaseMLP, BaseMLP_2
from models.utils.ctrf import BaseTransformer
from models.utils.scatter import scatter_batch
from models.utils.frame_building_sc import build_paf_frame
from models.utils.edge_index_gen import get_g2b_indices, get_intergraph_edge_index
from models.utils.GIL import GIL

class GeoMix(nn.Module):
    def __init__(self, num_layer, node_input_dim, edge_attr_dim, hidden_dim, salt_embed_dim, 
                num_mlp_layer, num_atten_layer, num_atten_head, 
                num_gil_layer=3, seq_len=8, 
                activation=nn.SiLU(), encoder='EGNN', device='cpu', if_salt_embed=True,
                att_dropout=0.0, att_temp=2.0):
        super(GeoMix, self).__init__()
        self.c = 1
        self.d = hidden_dim
        self.num_layer = num_layer
        self.num_gil_layer = num_gil_layer
        self.seq_len = seq_len
        self.noisy_sigma = 0.3
        self.act = activation
        self.device = device
        self.if_salt_embed = if_salt_embed

        if encoder == 'EGNN':
            self.enc_solv = EGNN(
                num_layer=num_layer,
                node_input_dim=node_input_dim,
                edge_attr_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                device=device
            )
            self.enc_salt = EGNN(
                num_layer=num_layer,
                node_input_dim=node_input_dim,
                edge_attr_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                device=device
            )
        elif encoder == 'TFN':
            self.enc_solv = TFN(
                num_layer=num_layer,
                node_input_dim=node_input_dim,
                edge_attr_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                require_vel=False,
                device=device
            )
            self.enc_salt = TFN(
                num_layer=num_layer,
                node_input_dim=node_input_dim,
                edge_attr_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                require_vel=False,
                device=device
            )

        self.GIL = nn.Sequential()
        for _ in range(num_gil_layer):
            layer = GIL(self.d, self.act, self.device)
            self.GIL.append(layer)

        self.att = BaseTransformer(hidden_dim, hidden_dim, hidden_dim, 
                            num_layer=num_atten_layer, num_heads=num_atten_head, 
                            activation=nn.SiLU(), residual=False, last_act=False, 
                            max_seq_len=self.seq_len,
                            attn_dropout=att_dropout,
                            attn_temperature=att_temp
                            )

        self.MLP_1 = BaseMLP_2(self.d+3, self.d+3, self.d+3, num_layer=num_mlp_layer, 
                               activation=activation, residual=True, last_act=True)
        self.readout_mlp = BaseMLP_2(self.d+3, self.d, 1, activation=nn.SiLU(), last_act=False)

        self.to(device)


    def forward(self, batch, salt_batch, n2g_indices, n2b_indices, g2b_indices, T, c, salt_one_hot):
        x_atom_precise = torch.cat([batch.pos, salt_batch.pos], dim=0)
        x_atom_noisy = x_atom_precise + torch.randn_like(x_atom_precise) * self.noisy_sigma
        F_graph = build_paf_frame(x_atom_noisy, n2g_indices).detach()
        F = F_graph[n2g_indices].detach()

        h_atom_solv = batch.x
        h_atom_salt = salt_batch.x
        x_atom_solv = x_atom_precise[:h_atom_solv.shape[0], ...]
        x_atom_salt = x_atom_precise[h_atom_solv.shape[0]:, ...]

        x_atom = torch.cat([x_atom_solv, x_atom_salt], dim=0)
        v_atom = torch.zeros_like(x_atom)
        edge_index = torch.cat([batch.edge_index, salt_batch.edge_index + len(batch.x)], dim=1)
        inter_edge_index = get_intergraph_edge_index(n2g_indices, n2b_indices)

        h_atom_solv_cat = torch.cat([h_atom_solv, h_atom_solv], dim=0)
        x_atom_solv_noisy = x_atom_solv + torch.randn_like(x_atom_solv) * self.noisy_sigma
        x_atom_solv_cat = torch.cat([x_atom_solv, x_atom_solv_noisy], dim=0)
        edge_index_solv_cat = torch.cat([batch.edge_index, batch.edge_index+x_atom_solv.shape[0]], dim=1)
        edge_attr_solv_cat = torch.cat([batch.edge_attr, batch.edge_attr], dim=0)
        h_atom_salt_cat = torch.cat([h_atom_salt, h_atom_salt], dim=0)
        x_atom_salt_noisy = x_atom_salt + torch.randn_like(x_atom_salt) * self.noisy_sigma
        x_atom_salt_cat = torch.cat([x_atom_salt, x_atom_salt_noisy], dim=0)
        edge_index_salt_cat = torch.cat([salt_batch.edge_index, salt_batch.edge_index+x_atom_salt.shape[0]], dim=1)
        edge_attr_salt_cat = torch.cat([salt_batch.edge_attr, salt_batch.edge_attr], dim=0)

        x_atom_solv_cat = x_atom_solv_cat.squeeze(1)
        x_atom_salt_cat = x_atom_salt_cat.squeeze(1)

        h_atom_solv_cat, x_atom_solv_cat = self.enc_solv(h_atom_solv_cat, x_atom_solv_cat, 
                                                         edge_index_solv_cat, edge_attr_solv_cat)
        h_atom_salt_cat, x_atom_salt_cat = self.enc_salt(h_atom_salt_cat, x_atom_salt_cat, 
                                                         edge_index_salt_cat, edge_attr_salt_cat)

        h_atom_solv = h_atom_solv_cat[:len(h_atom_solv)]
        x_atom_solv = x_atom_solv_cat[:len(x_atom_solv)]
        x_atom_solv_denoise = x_atom_solv_cat[len(x_atom_solv):]
        h_atom_salt = h_atom_salt_cat[:len(h_atom_salt)]
        x_atom_salt = x_atom_salt_cat[:len(x_atom_salt)]
        x_atom_salt_denoise = x_atom_salt_cat[len(x_atom_salt):]

        h_atom = torch.cat([h_atom_solv, h_atom_salt], dim=0)
        x_atom = torch.cat([x_atom_solv, x_atom_salt], dim=0)
        x_atom_denoise = torch.cat([x_atom_solv_denoise, x_atom_salt_denoise], dim=0)

        for i in range(self.num_gil_layer):
            h_atom, v_atom = self.GIL[i](h_atom, x_atom, v_atom, inter_edge_index, F)

        h_graph = scatter_mean(h_atom, n2g_indices, dim=0)
        batch_size = int(n2b_indices.max().item())+1
        h_graph_scattered = scatter_batch(h_graph, g2b_indices, batch_size, self.seq_len)
        h_sample = self.att(h_graph_scattered)

        feature_all = torch.cat([h_sample, T.unsqueeze(1), (1/T).unsqueeze(1), c.unsqueeze(1)], dim=1)
        feature_all = self.MLP_1(feature_all)
        res = self.readout_mlp(feature_all).squeeze(1)

        reg_denoise = torch.norm(x_atom_denoise - x_atom_precise, p='fro') / x_atom.shape[0]
        reg = reg_denoise * 128

        return res, reg
