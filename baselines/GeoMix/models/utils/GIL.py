import torch
from torch import nn
from torch_scatter import scatter

from models.utils.mlp import BaseMLP

class GIL(nn.Module):
    def __init__(self, hidden_dim, activation=nn.SiLU(), device='cpu'):
        super(GIL, self).__init__()
        self.d = hidden_dim
        self.device = device

        self._msg_mlp1 = BaseMLP(2*self.d+4, self.d, self.d, last_act=True)
        self._msg_mlp2 = BaseMLP(2*self.d+5, self.d, self.d, last_act=True)
        self._x_mlp = BaseMLP(self.d, self.d, 1)
        self._v_mlp = BaseMLP(self.d, self.d, 1)
        self._h_mlp = BaseMLP(2*self.d, self.d, self.d, last_act=True)
        self._9d_rot = BaseMLP(self.d, self.d, 9, last_act=True)
        self._t_mlp = BaseMLP(self.d, self.d, 3, last_act=True)

        self.to(device)

    def _msg(self, h, x, v, F, edge_index):
        row, col = edge_index
        h_i, h_j = h[row], h[col]
        x_i, x_j = x[row], x[col]
        v_i, v_j = v[row], v[col]
        F_i, F_j = F[row], F[col]

        x_i_norm = torch.norm(x_i, p=2, dim=-1, keepdim=True)
        x_j_norm = torch.norm(x_j, p=2, dim=-1, keepdim=True)
        v_i_norm = torch.norm(v_i, p=2, dim=-1, keepdim=True)
        v_j_norm = torch.norm(v_j, p=2, dim=-1, keepdim=True)

        msg_mc_input = torch.cat([h_i, h_j, x_i_norm, x_j_norm, v_i_norm, v_j_norm], dim=-1)
        msg_mc = self._msg_mlp1(msg_mc_input)

        rot_9d = self._9d_rot(msg_mc)
        rotmat = rot_9d.view(rot_9d.shape[0], 3, 3)
        t = self._t_mlp(msg_mc)
        t_norm = torch.norm(t, p=2, dim=-1, keepdim=True)

        msg_input = torch.cat([h_i, h_j, x_i_norm, x_j_norm, v_i_norm, v_j_norm, t_norm], dim=-1)
        msg = self._msg_mlp2(msg_input)

        msg_v = self._x_mlp(msg) * x_j + self._v_mlp(msg) * v_j
        msg_v = (rotmat @ F_j @ msg_v.unsqueeze(-1)) + t.unsqueeze(-1)
        msg_v = (F_i.transpose(-1, -2) @ msg_v).squeeze(-1)
        return msg, msg_v

    def _agg(self, msg, msg_v, edge_index, n):
        row, col = edge_index
        agg_h = scatter(src=msg, index=row, dim=0, dim_size=n, reduce='mean')
        agg_v = scatter(src=msg_v, index=row, dim=0, dim_size=n, reduce='mean')

        return agg_h, agg_v

    def _upd(self, h, v, agg_h, agg_v):
        v = v + agg_v
        h = torch.cat([h, agg_h], dim=-1)
        h = self._h_mlp(h)
        return h, v

    def forward(self, h, x, v, edge_index, F):
        """
        Args:
            h: [n_graph, d] molecular invariant features
            x: [n_graph, 3] molecular coordinates
            v: [n_graph, 3] molecular vector features
            edge_index
            F: [n_graph, 3, 3] molecular local frame
        """

        msg, msg_v = self._msg(h, x, v, F, edge_index)
        agg_h, agg_v = self._agg(msg, msg_v, edge_index, len(x))
        h, v = self._upd(h, v, agg_h, agg_v)
        return h, x
