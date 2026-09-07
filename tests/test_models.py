import pytest
import torch
from mamba_mis.models import MODEL_NAMES, build_model, logits_of
from mamba_mis.models.network import SpectralGate, AMFFB
from mamba_mis.metrics import segmentation_loss


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_all_models_forward(name):
    # PVT requires >=64 to accommodate its spatial reduction kernel.
    size = 64
    model = build_model({'name': name, 'image_size': size, 'backend': 'torch'}).eval()
    with torch.no_grad():
        y = logits_of(model(torch.randn(1, 3, size, size)))
    assert y.shape == (1, 1, size, size)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize('changes', [{'scan': 'ss2d'}, {'spectral': False}, {'ab': False},
    {'mffb': False}, {'ab': False, 'mffb': False}, {'block': 'cnn'}, {'block': 'vit'}])
def test_ablations_backward(changes):
    model = build_model(dict(name='mamba_mis_s', image_size=32, state_dim=2, backend='torch', **changes))
    x = torch.randn(2, 3, 32, 32)
    loss = segmentation_loss(model(x), torch.randint(0, 2, (2, 1, 32, 32)).float())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_spectral_identity_and_gradient():
    sg = SpectralGate(3, (5, 8))
    with torch.no_grad():
        sg.weight[..., 0].fill_(1); sg.weight[..., 1].zero_(); sg.bias.zero_()
    x = torch.randn(2, 5, 8, 3, requires_grad=True)
    y = sg(x)
    torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
    y.square().mean().backward()
    assert torch.isfinite(sg.weight.grad).all()


def test_rectangular_model():
    m = build_model(dict(name='mamba_mis_s', image_size=(32, 64), state_dim=2, backend='torch')).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 32, 64)).shape == (1, 1, 32, 64)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Full-resolution GPU baseline check')
@pytest.mark.parametrize('name',[n for n in MODEL_NAMES if not n.startswith('mamba_mis_')])
def test_full_resolution_baseline_cuda(name):
    model = build_model({'name':name,'image_size':256}).cuda().eval()
    with torch.inference_mode():
        out = logits_of(model(torch.randn(1,3,256,256,device='cuda')))
    assert out.shape == (1,1,256,256)
    assert torch.isfinite(out).all()
