"""Exact ZOH CUDA scan. Checkpoint every 32 tokens and recompute within backward.

Parallel over batch/channel and vectorized over states. B/C gradients use atomic
reductions across channels; select the torch backend for deterministic reductions.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(U, DT, A, B, C, D, Y, H, L: tl.constexpr, DIM: tl.constexpr,
             N: tl.constexpr, NS: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    batch, channel = pid // DIM, pid % DIM
    ns = tl.arange(0, NS)
    a = tl.load(A + channel * N + ns, ns < N, 0)
    d = tl.load(D + channel)
    state = tl.full((NS,), 0, tl.float32)
    for t in range(L):
        if t % CHUNK == 0:
            tl.store(H + ((t // CHUNK) * tl.num_programs(0) + pid) * N + ns, state, ns < N)
        off = (batch * L + t) * DIM + channel
        dt, u = tl.load(DT + off), tl.load(U + off)
        z = dt * a
        small = tl.abs(z) < 1e-3
        q = dt * tl.where(small, 1 + z / 2 + z * z / 6 + z * z * z / 24,
                          (tl.exp(z) - 1) / tl.where(small, 1., z))
        b = tl.load(B + (batch * L + t) * N + ns, ns < N, 0)
        c = tl.load(C + (batch * L + t) * N + ns, ns < N, 0)
        state = tl.exp(z) * state + q * b * u
        y = tl.sum(tl.where(ns < N, state * c, 0), 0) + d * u
        tl.store(Y + off, y)


@triton.jit
def _backward(U, DT, A, B, C, D, GY, H, SCRATCH, GU, GDT, GA, GB, GC, GD,
              L: tl.constexpr, DIM: tl.constexpr, N: tl.constexpr,
              NS: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    batch, channel = pid // DIM, pid % DIM
    ns = tl.arange(0, NS)
    a = tl.load(A + channel * N + ns, ns < N, 0)
    d = tl.load(D + channel)
    gh = tl.full((NS,), 0, tl.float32)
    ga = tl.full((NS,), 0, tl.float32)
    gd = tl.full((), 0, tl.float32)
    for rev in range(tl.cdiv(L, CHUNK)):
        ci = tl.cdiv(L, CHUNK) - 1 - rev
        start = ci * CHUNK
        state = tl.load(H + (ci * tl.num_programs(0) + pid) * N + ns, ns < N, 0)
        # Save h_(t-1), only one chunk of scratch per channel.
        for j in range(CHUNK):
            t = start + j
            if t < L:
                tl.store(SCRATCH + (pid * CHUNK + j) * N + ns, state, ns < N)
                off = (batch * L + t) * DIM + channel
                dt, u = tl.load(DT + off), tl.load(U + off)
                z = dt * a
                small = tl.abs(z) < 1e-3
                q = dt * tl.where(small, 1 + z / 2 + z*z / 6 + z*z*z / 24,
                                  (tl.exp(z)-1) / tl.where(small, 1., z))
                b = tl.load(B + (batch * L + t) * N + ns, ns < N, 0)
                state = tl.exp(z) * state + q * b * u
        for jr in range(CHUNK):
            j = CHUNK - 1 - jr
            t = start + j
            if t < L:
                off = (batch * L + t) * DIM + channel
                dt, u, gy = tl.load(DT + off), tl.load(U + off), tl.load(GY + off)
                b = tl.load(B + (batch * L + t) * N + ns, ns < N, 0)
                c = tl.load(C + (batch * L + t) * N + ns, ns < N, 0)
                prev = tl.load(SCRATCH + (pid * CHUNK + j) * N + ns, ns < N, 0)
                z = dt * a
                ez = tl.exp(z)
                small = tl.abs(z) < 1e-3
                q = dt * tl.where(small, 1 + z / 2 + z*z / 6 + z*z*z / 24,
                                  (ez-1) / tl.where(small, 1., z))
                qa = dt*dt * tl.where(tl.abs(z) < 0.02,
                    0.5 + z/3 + z*z/8 + z*z*z/30 + z*z*z*z/144,
                    (ez*(z-1)+1) / tl.where(tl.abs(z) < 0.02, 1., z*z))
                ht = ez * prev + q * b * u
                gh = gh + gy * c
                gu = tl.sum(tl.where(ns < N, gh*q*b, 0), 0) + gy*d
                gdt = tl.sum(tl.where(ns < N, gh*ez*(a*prev+b*u), 0), 0)
                ga += gh * (dt*ez*prev + qa*b*u)
                gd += gy*u
                tl.store(GU + off, gu)
                tl.store(GDT + off, gdt)
                tl.atomic_add(GB + (batch*L+t)*N+ns, gh*q*u, ns < N, sem="relaxed")
                tl.atomic_add(GC + (batch*L+t)*N+ns, gy*ht, ns < N, sem="relaxed")
                gh = gh * ez
    tl.atomic_add(GA + channel*N+ns, ga, ns < N, sem="relaxed")
    tl.atomic_add(GD + channel, gd, sem="relaxed")


class _ZOH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, a, b, c, d):
        batch, length, dim = u.shape
        n, chunk = a.shape[1], 32
        h = torch.empty((triton.cdiv(length, chunk), batch*dim, n), device=u.device)
        y = torch.empty_like(u)
        _forward[(batch*dim,)](u, dt, a, b, c, d, y, h, length, dim, n,
                              triton.next_power_of_2(n), chunk, num_warps=4)
        ctx.save_for_backward(u, dt, a, b, c, d, h)
        return y

    @staticmethod
    def backward(ctx, gy):
        u, dt, a, b, c, d, h = ctx.saved_tensors
        batch, length, dim = u.shape
        n, chunk = a.shape[1], 32
        grads = [torch.zeros_like(x) for x in (u, dt, a, b, c, d)]
        scratch = torch.empty((batch*dim, chunk, n), device=u.device)
        _backward[(batch*dim,)](u, dt, a, b, c, d, gy.contiguous(), h, scratch,
                               *grads, length, dim, n, triton.next_power_of_2(n),
                               chunk, num_warps=4)
        return tuple(grads)


def zoh_triton(u, delta, a, b, c, d):
    if not u.is_cuda:
        raise ValueError("Triton backend requires CUDA; use backend='torch' on CPU")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Triton scan uses atomic gradient reductions; use backend='torch' for strict determinism")
    return _ZOH.apply(*[x.float().contiguous() for x in (u, delta, a, b, c, d)])
