import copy
import json
import pytest
import torch
from mamba_mis.config import DEFAULT, merge
from mamba_mis import engine
from mamba_mis.models import build_model
from mamba_mis.metrics import segmentation_loss
from test_data_metrics import fixture_rows


def test_synthetic_train_resume_evaluate_and_predict(tmp_path, monkeypatch):
    _, manifest = fixture_rows(tmp_path)
    config = merge(DEFAULT, {'model':{'name':'mamba_mis_s','image_size':32,'block':'cnn',
                                     'ab':False,'mffb':False,'backend':'torch'},
        'data':{'root':str(tmp_path),'manifest':str(manifest)},
        'train':{'epochs':2,'effective_batch_size':4,'micro_batch_size':2,'workers':0,'amp':'no'},
        'output':str(tmp_path/'resumed')})
    original_train = engine.epoch_train
    calls = 0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError('Synthetic interruption after saved epoch')
        return original_train(*args,**kwargs)
    monkeypatch.setattr(engine,'epoch_train',interrupt)
    with pytest.raises(InterruptedError):
        engine.train(config)
    monkeypatch.setattr(engine,'epoch_train',original_train)
    engine.train(config,resume=tmp_path/'resumed/last.pt')
    reference = copy.deepcopy(config); reference['output'] = str(tmp_path/'continuous')
    engine.train(reference)
    a = torch.load(tmp_path/'resumed/last.pt', weights_only=False)
    b = torch.load(tmp_path/'continuous/last.pt', weights_only=False)
    for k in a['model']:
        torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
    assert a['scheduler'] == b['scheduler']
    assert a['epoch'] == 1
    report = engine.evaluate(tmp_path/'resumed/best.pt',tmp_path/'report.json')
    assert report['results']['synthetic']['samples'] == 2
    report = engine.evaluate(tmp_path/'resumed/best.pt',tmp_path/'original.json',original=True)
    assert report['resolution'] == 'original'
    from mamba_mis.analysis import predict, visualize, summarize
    predict(tmp_path/'resumed/best.pt',[tmp_path/'test0.png'],tmp_path/'prediction')
    assert (tmp_path/'prediction/test0_mask.png').exists()
    visualize(tmp_path/'resumed/best.pt',tmp_path/'visual',limit=1)
    assert list((tmp_path/'visual').glob('*.png'))
    summary = summarize([tmp_path/'report.json'],tmp_path/'summary.json')
    assert summary[0]['metrics']['macro']['dice']['sample_std'] is None
    from mamba_mis.analysis import compare, profile, plot_comparison
    compare([tmp_path/'resumed/best.pt',tmp_path/'continuous/best.pt'],tmp_path/'compare',limit=1)
    assert (tmp_path/'compare/comparison.png').exists()
    profile(config,tmp_path/'profile.json',warmup=0,repeats=1,device_name='cpu')
    plot_comparison(tmp_path/'summary.json',[tmp_path/'profile.json'],'synthetic',tmp_path/'scatter.png',resolution='32')
    assert (tmp_path/'scatter.png').exists()


def test_accumulation_matches_full_batch():
    # Use a non-BN model; BN intentionally sees the physical micro-batch.
    torch.manual_seed(42)
    a = torch.nn.Conv2d(3,1,1)
    b = copy.deepcopy(a)
    inputs = torch.randn(5,3,4,4); targets = torch.randint(0,2,(5,1,4,4)).float()
    oa, ob = torch.optim.SGD(a.parameters(),lr=.1), torch.optim.SGD(b.parameters(),lr=.1)
    segmentation_loss(a(inputs),targets).backward(); oa.step()
    from torch.utils.data import DataLoader
    loader = DataLoader([{'image':x,'mask':y} for x,y in zip(inputs,targets)],batch_size=2)
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    engine.epoch_train(b,loader,ob,scaler,torch.device('cpu'),3,{'amp':'no','aux_weight':0.,'grad_clip':None})
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)


def test_experiment_matrix():
    from pathlib import Path
    from mamba_mis.experiments import experiments
    root = Path(__file__).resolve().parents[1]/'configs'
    jobs = experiments('all',root)
    assert len(jobs) == 138
    assert len({x['output'] for x in jobs}) == 138
    from mamba_mis.config import load_config
    for path in root.rglob('*.yaml'):
        config = load_config(path)
        assert config['train']['epochs'] == 300


def test_analysis_features_on_synthetic_checkpoint(tmp_path):
    from mamba_mis.data import compute_stats
    from mamba_mis.config import sha256
    from mamba_mis.analysis import visualize, compare
    rows,manifest = fixture_rows(tmp_path)
    config = merge(DEFAULT,{'model':{'name':'mamba_mis_s','image_size':32,'state_dim':2,
                                     'backend':'torch','use_checkpoint':False},
                            'data':{'root':str(tmp_path),'manifest':str(manifest)}})
    model = build_model(config['model'])
    checkpoint = tmp_path/'synthetic_untrained.pt'
    engine.atomic_save(checkpoint,{'model':model.state_dict(),'config':config,
        'stats':compute_stats(rows,tmp_path,32),'manifest_sha256':sha256(manifest)})
    for kind in ['spectral','bridge','saliency','erf']:
        visualize(checkpoint,tmp_path/kind,kind=kind,limit=1)
        assert list((tmp_path/kind).glob('*.png'))
    compare([checkpoint],tmp_path/'compare_erf',kind='erf',limit=1)
    assert (tmp_path/'compare_erf/comparison.png').exists()


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA')
@pytest.mark.parametrize('variant',['s','b','l'])
def test_full_resolution_gpu_backward(variant):
    model = build_model(dict(name=f'mamba_mis_{variant}',image_size=256,use_checkpoint=True)).cuda()
    x = torch.randn(1,3,256,256,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss = segmentation_loss(model(x),torch.zeros(1,1,256,256,device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
