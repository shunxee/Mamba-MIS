"""Diagonal selective SSM, exact zero-order hold; tensors are B,L,D and B,L,N."""
import torch
from torch import nn
from torch.nn import functional as F


def zoh_reference(u, delta, a, b, c, d):
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, a, b, c, d = [v.to(dtype) for v in (u, delta, a, b, c, d)]
    h = u.new_zeros(u.shape[0], u.shape[2], a.shape[1])
    outputs = []
    for t in range(u.shape[1]):
        z = delta[:, t, :, None] * a
        # expm1(z)/z has an analytic limit at zero. A is negative by construction.
        small = z.abs() < 1e-4
        safe_z = torch.where(small, torch.ones_like(z), z)
        phi = torch.where(small, 1 + z / 2 + z.square() / 6, torch.expm1(z) / safe_z)
        q = delta[:, t, :, None] * phi
        h = z.exp() * h + q * b[:, t, None, :] * u[:, t, :, None]
        outputs.append((h * c[:, t, None, :]).sum(-1) + d * u[:, t])
    return torch.stack(outputs, 1)


def selective_zoh(u, delta, a, b, c, d, backend="auto"):
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"Unknown scan backend: {backend}")
    if backend == "triton" or (backend == "auto" and u.is_cuda):
        from .triton_scan import zoh_triton
        return zoh_triton(u, delta, a, b, c, d)
    return zoh_reference(u, delta, a, b, c, d)


class S6(nn.Module):
    def __init__(self, dim, state_dim=32, backend="auto"):
        super().__init__()
        self.dim, self.state_dim, self.backend = dim, state_dim, backend
        self.delta = nn.Linear(dim, dim)
        self.bc = nn.Linear(dim, state_dim * 2, bias=False)
        self.a_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(dim, 1))
        self.d = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.delta.bias, -4.0)

    def forward(self, x):
        b, c = self.bc(x).chunk(2, -1)
        delta = F.softplus(self.delta(x).float())
        return selective_zoh(x, delta, -self.a_log.float().exp(), b, c, self.d,
                             self.backend).to(x.dtype)


def scan_sequences(x, four=False):
    """BHWC -> list of BLC sequences; vertical order is explicitly inverted later."""
    rows = x.flatten(1, 2)
    cols = x.transpose(1, 2).flatten(1, 2)
    return [rows, cols, rows.flip(1), cols.flip(1)] if four else [rows, cols]


def restore_sequences(ys, height, width):
    result = []
    for i, y in enumerate(ys):
        if i >= 2:
            y = y.flip(1)
        if i % 2:
            y = y.reshape(y.shape[0], width, height, -1).transpose(1, 2)
        else:
            y = y.reshape(y.shape[0], height, width, -1)
        result.append(y)
    return result


class BFSS(nn.Module):
    def __init__(self, dim, state_dim=32, backend="auto", scan="bfss"):
        super().__init__()
        if scan not in {"bfss", "ss2d"}:
            raise ValueError(scan)
        self.four = scan == "ss2d"
        self.scans = nn.ModuleList([S6(dim, state_dim, backend) for _ in range(4 if self.four else 2)])
        self.fusion = (nn.Identity() if self.four else nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim)))

    def forward(self, x):
        ys = [op(seq) for op, seq in zip(self.scans, scan_sequences(x, self.four))]
        ys = restore_sequences(ys, x.shape[1], x.shape[2])
        return self.fusion(sum(ys) if self.four else torch.cat(ys, -1))
