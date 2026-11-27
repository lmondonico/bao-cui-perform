import torch
from torch import nn
from torch_scatter import scatter_mean

from models.modules.EGNN import EGNN
from models.utils.mlp import BaseMLP, BaseMLP_2

class EGNN_linear(nn.Module):
    def __init__(self, num_layer, node_input_dim, edge_attr_dim, hidden_dim, salt_embed_dim, 
                 num_mlp_layer, activation=nn.SiLU(), device='cpu', if_salt_embed=True):
        super(EGNN_linear, self).__init__()
        self.device = device
        self.if_salt_embed = if_salt_embed

        self.EGNN = EGNN(num_layer, node_input_dim, edge_attr_dim, hidden_dim, 
                         activation=activation, device=device)
        if if_salt_embed:
            self.EGNN_salt = EGNN(num_layer, node_input_dim, edge_attr_dim, hidden_dim, 
                             activation=activation, device=device)
        else:
            self.embedding_salt = nn.Linear(salt_embed_dim, hidden_dim)
        self.MLP_1 = BaseMLP_2(hidden_dim+2, hidden_dim+2, hidden_dim+2, num_layer=num_mlp_layer, 
                               activation=activation, residual=True, last_act=True)
        self.readout_MLP = BaseMLP(hidden_dim+2, hidden_dim, 1, 
                                   activation=nn.Softplus(), residual=False, last_act=True)
        

    def forward(self, batch, salt_batch, n2g_indices, n2b_indices, g2b_indices, T, c, salt_one_hot):
        """
        Args:
            batch: a PyG batch with several rows of data, each row contains more than one solvents. 
                Tensor sample_indices is a attribute of batch, show which row the subgraph from
            salt_batch: a PyG batch like batch above, but contains one or more salts
            T: temperature
            c: concentration
            salt_one_hot: salt one-hot embeddings
        """
        h_atom_solv, _ = self.EGNN(batch.x, batch.pos, batch.edge_index, batch.edge_attr)

        if self.if_salt_embed:
            h_atom_salt, _ = self.EGNN_salt(salt_batch.x, salt_batch.pos, salt_batch.edge_index, salt_batch.edge_attr)
        else:
            h_atom_salt = self.embedding_salt(salt_one_hot)
        h_atom = torch.cat([h_atom_solv, h_atom_salt], 0)
        h_sample = scatter_mean(h_atom, n2b_indices, dim=0)
        feature_all = torch.cat((h_sample, T.unsqueeze(-1), c.unsqueeze(-1)), -1)

        feature_all = self.MLP_1(feature_all)
        res = self.readout_MLP(feature_all).squeeze(1)
        return res