import torch
from torch_scatter import scatter

def get_g2b_indices(n2g, n2b):
    n_graph = n2g.max().item() + 1
    g2b = scatter(n2b, n2g, dim=0, dim_size=n_graph, reduce='mean').to(torch.long)
    return g2b

def get_intergraph_edge_index(n2g, n2b):
    same_batch = n2b.unsqueeze(0) == n2b.unsqueeze(1)
    diff_graph = n2g.unsqueeze(0) != n2g.unsqueeze(1)

    mask = same_batch & diff_graph
    mask = torch.triu(mask, diagonal=1)
    row, col = mask.nonzero(as_tuple=True)

    edge_index = torch.cat([
        torch.stack([row, col], dim=0),
        torch.stack([col, row], dim=0),
    ], dim=1)
    return edge_index

def get_cpl_graph_edge_index(g2b):
    device = g2b.device
    batch_ids = torch.unique(g2b)
    edge_list = []
    
    for b in batch_ids:
        mask = (g2b == b)
        nodes = mask.nonzero(as_tuple=True)[0]
        nb = nodes.shape[0]
        if nb < 2:
            continue

        rows = nodes.repeat_interleave(nb)
        cols = nodes.repeat(nb)

        self_loop_mask = rows != cols
        rows = rows[self_loop_mask]
        cols = cols[self_loop_mask]
        
        edge_list.append(torch.stack([rows, cols], dim=0))
    
    if edge_list:
        edge_index = torch.cat(edge_list, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return edge_index