from functools import partial

import torch
from torch import nn
from torch_scatter import scatter

from models.utils.mlp import BaseMLP

class EGNN_Layer(nn.Module):
    def __init__(self, edge_attr_dim, hidden_dim, activation=nn.SiLU()):
        super(EGNN_Layer, self).__init__()
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_msg = MLP(input_dim=2 * hidden_dim + edge_attr_dim + 1, output_dim=hidden_dim, last_act=True)
        self.mlp_pos = MLP(input_dim=hidden_dim, output_dim=1)
        self.mlp_node_feat = MLP(input_dim=hidden_dim + hidden_dim, output_dim=hidden_dim)

    def forward(self, node_feat, node_pos, edge_index, edge_attr):
        msg, diff_pos = self.Msg(edge_index, edge_attr, node_feat, node_pos)
        msg_agg, pos_agg = self.Agg(edge_index, node_feat.size(0), msg, diff_pos)
        node_feat, node_pos = self.Upd(node_feat, node_pos, msg_agg, pos_agg)
        return node_feat, node_pos
    
    def Msg(self, edge_index, edge_attr, node_feat, node_pos):
        row, col = edge_index
        diff_pos = (node_pos[row] - node_pos[col])/(torch.norm(node_pos[row] - node_pos[col], dim=-1, keepdim=True) + 1)
        dist = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1) ** 2
        
        msg = torch.cat([node_feat[row], node_feat[col], edge_attr.unsqueeze(-1), dist], dim=-1)
        msg = self.mlp_msg(msg)
        diff_pos = diff_pos * self.mlp_pos(msg)
        return msg, diff_pos
    
    def Agg(self, edge_index, dim_size, msg, diff_pos):
        row, col = edge_index
        pos_agg = scatter(src=diff_pos, index=row, dim=0, dim_size=dim_size, reduce='mean')
        msg_agg = scatter(src=msg, index=row, dim=0, dim_size=dim_size, reduce='mean')
        return msg_agg, pos_agg
    
    def Upd(self, node_feat, node_pos, msg_agg, pos_agg):
        node_pos = node_pos + pos_agg
        node_feat = torch.cat([node_feat, msg_agg], dim=-1)
        node_feat = self.mlp_node_feat(node_feat)
        return node_feat, node_pos


class EGNN(nn.Module):
    def __init__(self, num_layer, node_input_dim, edge_attr_dim, hidden_dim, activation=nn.SiLU(), device='cpu'):
        super(EGNN, self).__init__()
        self.num_layer = num_layer
        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            layer = EGNN_Layer(edge_attr_dim, hidden_dim, activation=activation)
            self.layers.append(layer)
        self.to(device)

    def forward(self, node_feat, node_pos, edge_index, edge_attr):
        node_feat = self.embedding(node_feat)
        for layer in self.layers:
            node_feat, node_pos = layer(node_feat, node_pos, edge_index, edge_attr)
        return node_feat, node_pos
