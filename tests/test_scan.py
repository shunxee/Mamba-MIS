import pytest
import torch
from mamba_mis.models.scan import scan_sequences, restore_sequences, zoh_reference


def inputs(device='cpu', dtype=torch.float64):
    torch.manual_seed(11)
    return [x.to(device=device, dtype=dtype).requires_grad_() for x in [
        torch.randn(2, 9, 3), torch.rand(2, 9, 3)*.2,
        -torch.rand(3, 4)*3-.1, torch.randn(2, 9, 4), torch.randn(2, 9, 4), torch.randn(3)]]


def test_scan_coordinate_inverse():
    x = torch.arange(2*3*5*4).reshape(2, 3, 5, 4)
    for y in restore_sequences(scan_sequences(x, True), 3, 5):
        assert torch.equal(x, y)


def test_zoh_gradcheck():
    assert torch.autograd.gradcheck(zoh_reference, tuple(inputs()), atol=1e-5)


def test_zoh_scalar_analytic_and_small_a():
    u = torch.ones(1, 5, 1, dtype=torch.float64)
    for a in [-2., -1e-10]:
        y = zoh_reference(u, u*.1, torch.tensor([[a]], dtype=torch.float64), u, u, torch.zeros(1))
        expected = torch.expm1(torch.arange(1, 6, dtype=torch.float64)*.1*a)/a
        torch.testing.assert_close(y.flatten(), expected, atol=1e-8, rtol=1e-7)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('length', [9, 32, 33, 71])
def test_triton_forward_and_gradients(length):
    from mamba_mis.models.triton_scan import zoh_triton
    a = inputs('cuda', torch.float32)
    a = [x.detach().repeat(1, (length+8)//9, 1)[:, :length].contiguous().requires_grad_()
         if x.ndim == 3 else x for x in a]
    b = [x.detach().clone().requires_grad_() for x in a]
    ya, yb = zoh_reference(*a), zoh_triton(*b)
    torch.testing.assert_close(ya, yb, atol=2e-4, rtol=2e-3)
    g = torch.randn_like(ya)
    ga, gb = torch.autograd.grad(ya, a, g), torch.autograd.grad(yb, b, g)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=5e-4, rtol=5e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('grouped',[False, True])
def test_original_baseline_cuda_scan_against_reference(grouped):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_mis.models.baseline_ops import selective_scan_ref
    torch.manual_seed(8)
    shape = (2,2,3,11) if grouped else (2,3,11)
    values = [torch.randn(2,4,11,device='cuda'), torch.rand(2,4,11,device='cuda'),
              -torch.rand(4,3,device='cuda'), torch.randn(shape,device='cuda'),
              torch.randn(shape,device='cuda'), torch.ones(4,device='cuda')]
    a = [v.requires_grad_() for v in values]
    b = [v.detach().clone().requires_grad_() for v in values]
    x = selective_scan_fn(*a,delta_softplus=True)
    y = selective_scan_ref(*b,delta_softplus=True)
    torch.testing.assert_close(x,y,atol=2e-4,rtol=2e-3)
    grad = torch.randn_like(x)
    ga,gb = torch.autograd.grad(x,a,grad),torch.autograd.grad(y,b,grad)
    for p,q in zip(ga,gb):
        torch.testing.assert_close(p,q,atol=5e-4,rtol=5e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_portable_mamba1_matches_official():
    from mamba_ssm import Mamba
    from mamba_mis.models.baseline_ops import Mamba1
    torch.manual_seed(42)
    official = Mamba(d_model=8,d_state=4,d_conv=4,expand=2,use_fast_path=False).cuda().eval()
    portable = Mamba1(d_model=8,d_state=4,d_conv=4,expand=2).cuda().eval()
    portable.load_state_dict(official.state_dict(),strict=True)
    x = torch.randn(2,19,8,device='cuda')
    torch.testing.assert_close(official(x),portable(x),atol=2e-4,rtol=2e-3)
