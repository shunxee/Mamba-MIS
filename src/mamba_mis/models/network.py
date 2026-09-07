"""Mamba-MIS encoder, decoder, spectral gates and multi-scale feature fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .scan import BFSS

DEPTHS = {"s": ([1, 2, 2, 2], [2, 2, 2, 1, 1]),
          "b": ([2, 2, 2, 2], [2, 2, 2, 2, 2]),
          "l": ([3, 4, 4, 9], [9, 4, 4, 3, 3])}


class SpectralGate(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        h, w = size
        self.size = tuple(size)
        shape = (h, w // 2 + 1, dim, 2)
        self.weight = nn.Parameter(torch.randn(shape) * .02)
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        if tuple(x.shape[1:3]) != self.size:
            raise ValueError(f"Spectral grid expects {self.size}, got {x.shape[1:3]}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
            y = spectrum * torch.view_as_complex(self.weight.float()) + torch.view_as_complex(self.bias.float())
            y = torch.fft.irfft2(y, s=self.size, dim=(1, 2), norm="ortho")
        return y.to(x.dtype)


class BSSS(nn.Module):
    def __init__(self, dim, size, state_dim=32, backend="auto", scan="bfss", spectral=True):
        super().__init__()
        self.spectral = spectral
        self.parts = [dim // 3, 2*dim // 3 - dim // 3, dim - 2*dim // 3]
        self.norms = nn.ModuleList([nn.LayerNorm(k) for k in self.parts[:3 if spectral else 2]])
        self.projs = nn.ModuleList([nn.Linear(k, dim) for k in self.parts[:3 if spectral else 2]])
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.scan = BFSS(dim, state_dim, backend, scan)
        if spectral:
            self.sg = SpectralGate(dim, size)
            self.sg_norm = nn.LayerNorm(dim)
        self.fusion = nn.Linear(dim * (2 if spectral else 1), dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        chunks = x.split(self.parts, -1)
        branches = [p(n(v)) for p, n, v in zip(self.projs, self.norms, chunks)]
        gate = F.silu(branches[0])
        spatial = self.dw(branches[1].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        spatial = self.scan(F.silu(spatial))
        fused = (torch.cat([spatial, F.silu(self.sg_norm(self.sg(branches[2])))], -1)
                 if self.spectral else spatial)
        return x + self.out(gate * self.fusion(fused))


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):
        return self.block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class ViTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x):
        b, h, w, d = x.shape
        tokens = x.reshape(b, h*w, d)
        q, k, v = self.qkv(self.norm1(tokens)).reshape(b, h*w, 3, 4, d//4).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, h*w, d)
        tokens = tokens + self.out(y)
        return (tokens + self.mlp(self.norm2(tokens))).reshape(b, h, w, d)


class Stage(nn.Module):
    def __init__(self, dim, size, depth, block, use_checkpoint, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if block not in {"bsss", "cnn", "vit"}:
            raise ValueError(block)
        self.blocks = nn.ModuleList([BSSS(dim, size, **kwargs) if block == "bsss" else
                                    ConvBlock(dim) if block == "cnn" else ViTBlock(dim) for _ in range(depth)])

    def forward(self, x):
        for layer in self.blocks:
            x = checkpoint(layer, x, use_reentrant=False) if self.use_checkpoint and self.training else layer(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(4*dim), nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        return self.proj(self.norm(torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                                              x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)))


class PatchExpand(nn.Module):
    def __init__(self, dim, out_dim, factor):
        super().__init__()
        self.factor, self.out_dim = factor, out_dim
        self.proj, self.norm = nn.Linear(dim, out_dim*factor*factor, bias=False), nn.LayerNorm(out_dim)

    def forward(self, x):
        b, h, w, _ = x.shape
        f, d = self.factor, self.out_dim
        x = self.proj(x).reshape(b, h, w, f, f, d).permute(0, 1, 3, 2, 4, 5)
        return self.norm(x.reshape(b, h*f, w*f, d))


class AttentionBridge(nn.Module):
    def __init__(self, dim, state_dim, backend, scan):
        super().__init__()
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_scan = BFSS(1, state_dim, backend, scan)
        hidden = max(1, dim // 16)
        self.channel_in = nn.Linear(dim, hidden, bias=False)
        self.channel_scan = BFSS(hidden, state_dim, backend, scan)
        self.channel_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        sa = torch.cat([x.mean(-1, keepdim=True), x.amax(-1, keepdim=True)], -1)
        sa = self.spatial_conv(sa.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = x * self.spatial_scan(sa).sigmoid()
        pools = [y.mean((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)]
        ca = sum(self.channel_out(F.silu(self.channel_scan(self.channel_in(p)))) for p in pools)
        return x + y * ca.sigmoid()


class AMFFB(nn.Module):
    def __init__(self, dims, state_dim=32, backend="auto", scan="bfss", ab=True, mffb=True):
        super().__init__()
        self.ab, self.mffb = ab, mffb
        self.attentions = nn.ModuleList([AttentionBridge(d, state_dim, backend, scan) for d in dims]) if ab else None
        if mffb:
            self.reduce = nn.ModuleList([nn.Conv2d(d, 32, 1) for d in dims])
            self.smooth = nn.ModuleList([nn.ModuleList([nn.Conv2d(32, 32, 3, padding=1, groups=32)
                                                        for _ in dims]) for _ in dims])
            self.expand = nn.ModuleList([nn.Conv2d(32, d, 1) for d in dims])

    def forward(self, xs):
        ys = [a(x) for a, x in zip(self.attentions, xs)] if self.ab else xs
        if not self.mffb:
            return ys
        reduced = [r(y.permute(0, 3, 1, 2)) for r, y in zip(self.reduce, ys)]
        result = []
        for i, target in enumerate(reduced):
            product = torch.ones_like(target, dtype=torch.float32)
            for j, source in enumerate(reduced):
                size = target.shape[-2:]
                if j < i:
                    source = F.adaptive_avg_pool2d(source, size)
                elif j > i:
                    source = F.interpolate(source, size=size, mode="bilinear", align_corners=False)
                product = product * self.smooth[i][j](source).float()
            result.append(self.expand[i](product.to(target.dtype)).permute(0, 2, 3, 1))
        return result


class MambaMIS(nn.Module):
    def __init__(self, variant="b", image_size=256, state_dim=32, backend="auto", scan="bfss",
                 spectral=True, ab=True, mffb=True, block="bsss", use_checkpoint=False):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if any(s % 32 for s in self.image_size):
            raise ValueError("image_size must be divisible by 32 in each dimension")
        dims = [16, 32, 64, 128]
        enc, dec = DEPTHS[variant.lower()]
        sizes = [tuple(s // (4 * 2**i) for s in self.image_size) for i in range(4)]
        args = dict(state_dim=state_dim, backend=backend, scan=scan, spectral=spectral)
        self.embed, self.embed_norm = nn.Conv2d(3, 16, 4, stride=4), nn.LayerNorm(16)
        self.encoder = nn.ModuleList([Stage(d, s, dep, block, use_checkpoint, **args)
                                      for d, s, dep in zip(dims, sizes, enc)])
        self.merges = nn.ModuleList([PatchMerge(d) for d in dims[:-1]])
        self.bridge = AMFFB(dims, state_dim, backend, scan, ab, mffb)
        ddims, dsizes = dims[::-1] + [8], sizes[::-1] + [self.image_size]
        self.decoder = nn.ModuleList([Stage(d, s, dep, block, use_checkpoint, **args)
                                      for d, s, dep in zip(ddims, dsizes, dec)])
        self.expands = nn.ModuleList([PatchExpand(ddims[i], ddims[i+1], f)
                                      for i, f in enumerate([2, 2, 2, 4])])
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(f"Configured size {self.image_size}; received {tuple(x.shape[-2:])}")
        x = self.embed_norm(self.embed(x).permute(0, 2, 3, 1))
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            skips.append(x)
            if i < 3:
                x = self.merges[i](x)
        skips = self.bridge(skips)
        x = self.decoder[0](x + skips[3])
        for i, expand in enumerate(self.expands):
            x = expand(x)
            if i < 3:
                x = x + skips[2-i]
            x = self.decoder[i+1](x)
        return self.head(x.permute(0, 3, 1, 2))
