import torch
from torch import nn
from torch_scatter import scatter_mean

from models.modules.EGNN import EGNN
from models.utils.mlp import BaseMLP, BaseMLP_2
from models.utils.transformer import BaseTransformer
from models.utils.scatter import scatter_batch

class EGNN_atten(nn.Module):
    def __init__(self, num_layer, node_input_dim, edge_attr_dim, hidden_dim, 
                 num_mlp_layer, num_atten_layer, num_atten_head, 
                 seq_len=8, activation=nn.SiLU(), device='cpu'):
        super(EGNN_atten, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.EGNN = EGNN(num_layer, node_input_dim, edge_attr_dim, hidden_dim, 
                         activation=activation, device=device)
        self.EGNN_salt = EGNN(num_layer, node_input_dim, edge_attr_dim, hidden_dim, 
                         activation=activation, device=device)

        self.att = BaseTransformer(hidden_dim, hidden_dim, hidden_dim, 
                                   num_layer=num_atten_layer, num_heads=num_atten_head, 
                                   activation=nn.SiLU(), residual=False, last_act=False, 
                                   max_seq_len=self.seq_len)

        self.MLP_1 = BaseMLP_2(hidden_dim+2, hidden_dim+2, hidden_dim+2, num_layer=num_mlp_layer, 
                               activation=activation, residual=True, last_act=True)
        self.readout_MLP = BaseMLP(hidden_dim+2, hidden_dim, 1, 
                                   activation=nn.Softplus(), residual=False, last_act=True)


    def forward(self, batch, salt_batch, n2g_indices, n2b_indices, g2b_indices, T, c, salt_one_hot):
        """
        Args:
            batch: a PyG batch with several rows of data, each row contains more than one solvents. 
                Tensor sample_indices is a attribute of batch, show which row the subgraph from
            T: temperature
            c: concentration
            solute_emb: solute embeddings
        """
        h_atom_solv, _ = self.EGNN(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        h_atom_salt, _ = self.EGNN_salt(salt_batch.x, salt_batch.pos, salt_batch.edge_index, salt_batch.edge_attr)
        h_atom = torch.cat((h_atom_solv, h_atom_salt), 0)

        h_graph = scatter_mean(h_atom, n2g_indices, dim=0)
        batch_size = int(n2b_indices.max().item())+1

        h_sample = scatter_batch(h_graph, g2b_indices, batch_size, self.seq_len)

        h_sample = self.att(h_sample)
        feature_all = torch.cat((h_sample, T.unsqueeze(-1), c.unsqueeze(-1)), -1)
        
        feature_all = self.MLP_1(feature_all)
        res = self.readout_MLP(feature_all).squeeze(1)
        return res
