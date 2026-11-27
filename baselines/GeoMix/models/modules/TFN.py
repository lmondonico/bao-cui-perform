from typing import Optional
from functools import partial

import torch
from torch import nn, LongTensor, Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter
import e3nn

from models.utils.mace_modules.blocks import RadialEmbeddingBlock
from models.utils.mace_modules.irreps_tools import irreps2gate


class BaseMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        residual: bool = False, 
        last_act: bool = False,
    ) -> None:
        super(BaseMLP, self).__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Identity() if norm is None else norm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            nn.Identity() if norm is None else norm(output_dim),
            activation if last_act else nn.Identity()
        )

    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)
    

class TensorProductConvLayer(nn.Module):
    def __init__(
        self,
        in_irreps: e3nn.o3.Irreps,
        out_irreps: e3nn.o3.Irreps,
        sh_irreps: e3nn.o3.Irreps,
        hidden_dim: int = 64,
        edge_attr_dim: int = 2,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation, norm=norm)
        self.mlp_sh_coff = MLP(input_dim=edge_attr_dim, output_dim=self.tp.weight_numel)
        
    def forward(self, node_feat, edge_index, edge_attr, edge_sh):
        col, row = edge_index
        msg = self.tp(node_feat[col], edge_sh, self.mlp_sh_coff(edge_attr))
        agg = scatter(src=msg, index=row, dim=0, dim_size=node_feat.size(0), reduce='mean')
        node_feat = node_feat + agg
        return node_feat


class TFN(torch.nn.Module):
    """
    Tensor Field Network model from "Tensor Field Networks".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        require_vel: bool = True,
        num_layer: int = 4,
        hidden_dim: int = 64,
        irreps_channels: int = 8,
        node_input_dim: int = 2,
        edge_attr_dim: int = 2,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        device: str = 'cpu',
    ):
        super().__init__()
        
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="norm"
        )
        self.input_irreps = e3nn.o3.Irreps(f'{irreps_channels}x0e+1x1o') if require_vel else e3nn.o3.Irreps(f'{irreps_channels}x0e')
        self.hidden_irreps = (sh_irreps * irreps_channels).sort()[0].simplify()
        
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        
        self.embedding = nn.Linear(node_input_dim, irreps_channels)
        self.embedding2 = nn.Linear(self.hidden_irreps.dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(
                TensorProductConvLayer(
                    in_irreps=self.hidden_irreps,
                    out_irreps=self.hidden_irreps,
                    sh_irreps=sh_irreps,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=self.radial_embedding.out_dim + edge_attr_dim,
                    activation=activation,
                    norm=norm,
                )
            )
            
        self.get_delta_pos = e3nn.o3.Linear(self.hidden_irreps, '1x1o')

    
    def forward(self, node_feat, node_pos, edge_index, edge_attr):
        node_feat = self.embedding(node_feat)
        col, row = edge_index

        diff_pos = node_pos[col] - node_pos[row]
        dist = torch.norm(diff_pos, dim=1, keepdim=True)

        edge_attr = torch.cat([i for i in [
            edge_attr.unsqueeze(-1),
            self.radial_embedding(dist)
        ] if i is not None], dim=-1)
        edge_sh = self.spherical_harmonics(diff_pos)
        
        node_feat = F.pad(node_feat, (0, self.hidden_irreps.dim - self.input_irreps.dim))
        for layer in self.layers:
            node_feat = layer(node_feat, edge_index, edge_attr, edge_sh)
        node_pos = node_pos + self.get_delta_pos(node_feat)
        return self.embedding2(node_feat), node_pos