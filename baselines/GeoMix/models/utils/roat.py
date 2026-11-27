import torch

def rot_6D_to_mat(v):
    """
        Args:
            v: [..., 6]
        Returns:
            R: [..., 3, 3]
    """
    v = v.unsqueeze(-1)
    t1 = v[..., 0:3, :]
    t2 = v[..., 3:6, :]
    r1 = t1/torch.norm(t1, dim=-1, keepdim=True)
    r2 = t2 - (r1.transpose(-1, -2) @ t2) * r1
    r2 = r2/torch.norm(r2, dim=-1, keepdim=True)
    r3 = torch.cross(r1, r2, dim=-2)
    R = torch.cat([r1, r2, r3], dim=-1)
    return R

def rot_mat26D(R):
    pass