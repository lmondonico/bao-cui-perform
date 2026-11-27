import torch
import numpy as np
import itertools

def build_paf_frame(x, indices):
    """
    Args:
        x: [n_atom, 3] 原子坐标张量
        indices: [n_atom] 分子索引（每个元素表示原子所属的分子）
    Returns:
        basis: [n_graph, 3, 3] 分子局部基矩阵
    """
    n_atom, _ = x.shape
    device = x.device
    n_graph = indices.max().item() + 1

    x_flat = x.reshape(-1, 3)
    mol_idx = indices.view(-1, 1).reshape(-1)
    group_ids = mol_idx

    total_groups = n_graph
    counts = torch.zeros(total_groups, device=device).scatter_add_(
        0, group_ids, torch.ones_like(group_ids, dtype=torch.float32))
    counts = counts.clamp_min(1.0)

    sum_coords = torch.zeros(total_groups, 3, device=device).scatter_add_(
        0, group_ids.unsqueeze(1).expand(-1, 3), x_flat)
    means = sum_coords / counts.unsqueeze(1)

    x_centered = x_flat - means[group_ids]

    cov = torch.zeros(total_groups, 3, 3, device=device)
    for j in range(3):
        for k in range(3):
            prod = x_centered[:, j] * x_centered[:, k]
            cov[:, j, k].scatter_add_(0, group_ids, prod)
    cov = cov / counts.view(-1, 1, 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    sorted_idx = torch.argsort(eigvals, dim=1, descending=True)
    eigvecs = eigvecs.gather(2, sorted_idx.unsqueeze(1).expand(-1, 3, -1))

    L_bast, _ = get_optimal_bases(x_centered, group_ids, eigvecs)
    return L_bast

def get_optimal_bases(pos, indices, v):
    """
    pos: [n_atom, 3] float tensor
    indices: [n_atom] long tensor
    v: [n_graph, 3, 3] float tensor
    """
    n_atom, n_graph = pos.shape[0], v.shape[0]
    device = pos.device

    signs = torch.tensor(list(itertools.product([1, -1], repeat=3)), device=device)

    v_expand = v.unsqueeze(1).repeat(1, 8, 1, 1)
    sign_expand = signs.view(1, 8, 3, 1).expand(n_graph, -1, -1, 3)
    bases = v_expand * sign_expand

    bases_inv = torch.inverse(bases)

    bases_inv_per_point = bases_inv[indices]

    pos_expand = pos.unsqueeze(1).unsqueeze(-2)
    frac_all = torch.matmul(pos_expand, bases_inv_per_point).squeeze(-2)

    frac_sums = frac_all.sum(dim=-1)
    frac_sums_min, best_idx = frac_sums.min(dim=-1)

    frac_best = frac_all[torch.arange(n_atom), best_idx]

    best_idx_per_cell = torch.zeros(n_graph, dtype=torch.long, device=device)
    for i in range(n_graph):
        mask = (indices == i)
        if mask.any():
            selected_idx = best_idx[mask]
            vals, counts = selected_idx.unique(return_counts=True)
            best_idx_per_cell[i] = vals[counts.argmax()]

    L_best = bases[torch.arange(n_graph), best_idx_per_cell]

    return L_best, frac_best
