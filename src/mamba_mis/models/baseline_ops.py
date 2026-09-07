"""Mamba-1 baseline compatibility: original delta*B discretization, not ZOH.

CUDA dispatches to the official mamba-ssm extension. CPU reference supports
grouped B/C and the same weights, gating, convolution and parameterization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    original = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, length = u.shape
    def expand_bc(x):
        if x.ndim == 2:
            return x[None, :, :, None].expand(batch, -1, -1, length)
        if x.ndim == 3:
            return x[:, None].expand(-1, dim, -1, -1)
        return x.repeat_interleave(dim // x.shape[1], dim=1)
    B, C = expand_bc(B.float()), expand_bc(C.float())
    state = u.new_zeros(batch, dim, A.shape[1])
    ys = []
    for t in range(length):
        dt = delta[:, :, t, None]
        state = (dt * A.float()).exp() * state + dt * B[:, :, :, t] * u[:, :, t, None]
        y = (state * C[:, :, :, t]).sum(-1)
        if D is not None:
            y = y + D.float() * u[:, :, t]
        ys.append(y)
    y = torch.stack(ys, -1)
    if z is not None:
        y = y * F.silu(z.float())
    return (y.to(original), state) if return_last_state else y.to(original)


def selective_scan_fn(u, *args, **kwargs):
    if u.is_cuda:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as native
        return native(u, *args, **kwargs)
    return selective_scan_ref(u, *args, **kwargs)


class Mamba1(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=.001, dt_max=.1, dt_init="random", dt_scale=1., dt_init_floor=1e-4,
                 conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_inner, self.d_state, self.d_conv = int(expand*d_model), d_state, d_conv
        self.dt_rank = math.ceil(d_model/16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, 2*self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv-1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank+2*d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        bound = self.dt_rank**-.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -bound, bound)
        dt = torch.exp(torch.rand(self.d_inner)*(math.log(dt_max)-math.log(dt_min))+math.log(dt_min)).clamp_min(dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state+1).float().log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        u, z = self.in_proj(x).transpose(1, 2).chunk(2, 1)
        u = F.silu(self.conv1d(u)[..., :x.shape[1]])
        dt, b, c = self.x_proj(u.transpose(1, 2)).split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.linear(dt, self.dt_proj.weight).transpose(1, 2)
        y = selective_scan_fn(u, dt, -self.A_log.float().exp(), b.transpose(1, 2),
                              c.transpose(1, 2), self.D, z=z,
                              delta_bias=self.dt_proj.bias, delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))
