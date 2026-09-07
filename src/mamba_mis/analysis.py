"""Model profiling, feature visualization and experiment analysis."""
import json
import math
import statistics
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from .config import json_write
from .data import read_manifest, SegmentationDataset
from .engine import load_run, seed_all, runtime_metadata
from .models import build_model, logits_of
from .models.network import SpectralGate
from .models.scan import S6


def profile(config, output, warmup=5, repeats=20, device_name=None):
    if repeats < 1 or warmup < 0:
        raise ValueError('repeats >= 1 and warmup >= 0 required')
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed_all(42)
    model = build_model(config['model']).to(device).eval()
    size = config['model']['image_size']
    x = torch.randn(1, 3, size, size, device=device)
    counts = {'conv_linear_flops': 0, 'ssm_arithmetic_estimate': 0, 'ssm_exponentials': 0,
              'fft_flops_estimate': 0, 'attention_matmul_flops': 0}
    hooks = []
    def count(module, args, out):
        value = args[0]
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            counts['conv_linear_flops'] += out.numel()*math.prod(module.kernel_size)*module.in_channels//module.groups*2
        elif isinstance(module, nn.ConvTranspose2d):
            counts['conv_linear_flops'] += value.numel()*math.prod(module.kernel_size)*module.out_channels//module.groups*2
        elif isinstance(module, nn.Linear):
            counts['conv_linear_flops'] += out.numel()*module.in_features*2
        elif isinstance(module, SpectralGate):
            b, h, w, c = value.shape
            counts['fft_flops_estimate'] += b*c*(5*h*w*math.log2(h*w)+8*h*(w//2+1))
        elif isinstance(module, S6):
            n = value.numel()*module.state_dim
            counts['ssm_arithmetic_estimate'] += 16*n
            counts['ssm_exponentials'] += n
        elif module.__class__.__name__ == 'SS2D':
            # Original VMamba: four directional S6 paths; two branches projected first.
            n = value.shape[0]*value.shape[1]*value.shape[2]*module.d_inner*module.d_state*4
            counts['ssm_arithmetic_estimate'] += 9*n
            counts['ssm_exponentials'] += n
        elif module.__class__.__name__ == 'Mamba1':
            n = value.shape[0]*value.shape[1]*module.d_inner*module.d_state
            counts['ssm_arithmetic_estimate'] += 9*n
            counts['ssm_exponentials'] += n
        elif module.__class__.__name__ == 'ViTBlock':
            b, h, w, c = value.shape
            counts['attention_matmul_flops'] += 4*b*(h*w)**2*c
    for module in model.modules():
        hooks.append(module.register_forward_hook(count))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
    report = {'model': config['model'], 'input': list(x.shape), 'dtype': 'float32',
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'complex_parameter_convention': 'Two real scalars per complex parameter',
              'operations': counts, 'estimated_counted_gflops': sum(v for k,v in counts.items() if k != 'ssm_exponentials')/1e9,
              'operation_convention': 'Multiply/add each 1 FLOP. FFT: real transform estimate. SSM exp counted separately. Not a complete hardware FLOP count.',
              'not_counted': ['normalization', 'activation', 'pooling', 'interpolation', 'elementwise bridges/gates',
                              'baseline attention matrix products', 'linear projections implemented by einsum'],
              'latency_ms_median': statistics.median(timings), 'latency_ms_mean': statistics.mean(timings),
              'warmup': warmup, 'repeats': repeats,
              'peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device), 'runtime': runtime_metadata()}
    json_write(output, report)
    return report


def input_tensor(path, size, stats, device):
    with Image.open(path) as image:
        image = image.convert('RGB')
    tensor = torch.from_numpy(np.array(image.resize((size, size), Image.Resampling.BILINEAR), copy=True)).permute(2, 0, 1).float()/255
    tensor = (tensor-torch.tensor(stats['mean'])[:,None,None])/torch.tensor(stats['std'])[:,None,None]
    return tensor[None].to(device), image


def predict(checkpoint, inputs, output):
    if len({Path(p).stem for p in inputs}) != len(inputs):
        raise ValueError('Input basenames must be unique to prevent prediction files being overwritten')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_run(checkpoint, device)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        tensor, image = input_tensor(path, ckpt['config']['model']['image_size'], ckpt['stats'], device)
        with torch.inference_mode():
            logits = F.interpolate(logits_of(model(tensor)).float(), size=image.size[::-1], mode='bilinear', align_corners=False)
            probability = logits.sigmoid()[0, 0].cpu().numpy()
        stem = Path(path).stem
        np.save(output/f'{stem}_probability.npy', probability)
        mask = probability >= ckpt['config']['evaluation']['threshold']
        Image.fromarray((mask*255).astype(np.uint8)).save(output/f'{stem}_mask.png')
        rgb = np.asarray(image).astype(float)
        rgb[mask] = .55*rgb[mask]+.45*np.array([255, 30, 30])
        Image.fromarray(rgb.astype(np.uint8)).save(output/f'{stem}_overlay.png')


def visualize(checkpoint, output, kind='segmentation', ids=None, root=None, limit=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_run(checkpoint, device)
    config = ckpt['config']; seed_all(42)
    rows = sorted([r for r in read_manifest(config['data']['manifest']) if r['split'] == 'test'], key=lambda r:(r['dataset'], r['id']))
    if ids:
        requested = set(ids)
        rows = [r for r in rows if f"{r['dataset']}/{r['id']}" in requested]
        if len(rows) != len(requested):
            raise ValueError('Every --ids entry must match dataset/id in the test manifest')
    else:
        rows = rows[:limit]
    if not rows:
        raise ValueError('No visualization samples')
    ds = SegmentationDataset(rows, root or config['data']['root'], config['model']['image_size'], ckpt['stats'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selection = [f"{r['dataset']}/{r['id']}" for r in rows]
    json_write(output/'selection.json', {'samples': selection, 'kind': kind, 'seed':42,
                                        'normalization': 'shared range within each exported comparison'})
    if kind == 'erf':
        initial = build_model(config['model']).to(device).eval()
        maps = []
        for net in [initial, model]:
            gradients = []
            for item in ds:
                x = item['image'][None].to(device).requires_grad_()
                y = logits_of(net(x))
                g, = torch.autograd.grad(y[0,0,y.shape[-2]//2,y.shape[-1]//2], x)
                gradients.append(g.abs().sum(1)[0].detach().cpu().numpy())
            maps.append(np.mean(gradients, 0))
        scale = max(float(a.max()) for a in maps) or 1.
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        for ax, title, a in zip(axs, ['Random initialization', 'Checkpoint'], maps):
            ax.imshow(np.sqrt(a/scale), cmap='Greens', vmin=0, vmax=1); ax.set_title(title); ax.axis('off')
        np.savez(output/'erf_raw.npz', before=maps[0], after=maps[1])
        fig.tight_layout(); fig.savefig(output/'erf.png', dpi=180); plt.close(fig)
        return
    for i, item in enumerate(ds):
        x = item['image'][None].to(device)
        captures, hooks = {}, []
        if kind in {'spectral', 'bridge'}:
            if not config['model']['name'].startswith('mamba_mis_'):
                raise ValueError('spectral and bridge visualizations require Mamba-MIS')
            def capture(name):
                def hook(module, args, value):
                    captures[name] = [v.detach().float().cpu() for v in value] if isinstance(value, list) else value.detach().float().cpu()
                return hook
            for name, module in model.named_modules():
                if (kind == 'spectral' and isinstance(module, SpectralGate) and name.startswith(('decoder.3.', 'decoder.4.'))) or (kind == 'bridge' and name == 'bridge'):
                    hooks.append(module.register_forward_hook(capture(name)))
        try:
            if kind == 'saliency':
                x.requires_grad_()
                logits = logits_of(model(x))
                # Predicted foreground region selects the score; GT never enters the gradient target.
                region = (logits.detach().sigmoid() >= .5).float()
                if region.sum() == 0:
                    region = torch.ones_like(region)
                grad, = torch.autograd.grad((logits*region).sum()/region.sum(), x)
                panels = [('Input-gradient saliency', grad.abs().mean(1)[0].detach().cpu().numpy())]
            else:
                with torch.no_grad():
                    logits = logits_of(model(x))
                panels = []
        finally:
            for hook in hooks:
                hook.remove()
        base = item['image']*torch.tensor(ckpt['stats']['std'])[:,None,None]+torch.tensor(ckpt['stats']['mean'])[:,None,None]
        if kind == 'segmentation':
            p = logits.float().sigmoid()[0,0].detach().cpu().numpy()
            panels = [('Image', base.permute(1,2,0).clamp(0,1).numpy()), ('Ground truth', item['mask'][0].numpy()),
                      ('Probability', p), ('Prediction', p>=config['evaluation']['threshold'])]
        elif captures:
            for name, value in captures.items():
                values = value if isinstance(value, list) else [value]
                for j, v in enumerate(values):
                    panels.append((f'{name}/{j}', v[0].mean(-1).numpy()))
        if not panels:
            raise ValueError(f'No features captured for {kind}; check the ablation configuration')
        fig, axs = plt.subplots(1, len(panels), figsize=(4*len(panels),4), squeeze=False)
        bound = max(float(np.max(np.abs(p))) for _, p in panels) or 1.
        for ax, (title, p) in zip(axs[0], panels):
            ax.imshow(p, cmap='viridis', vmin=0 if kind in {'segmentation','saliency'} else -bound,
                      vmax=1 if kind == 'segmentation' else bound)
            ax.set_title(title, fontsize=9); ax.axis('off')
        tag = f'{i:03d}_{rows[i]["dataset"]}_{item["id"]}'
        np.savez(output/f'{tag}.npz', **{f'panel_{j}':p for j, (_,p) in enumerate(panels)})
        fig.tight_layout(); fig.savefig(output/f'{tag}.png', dpi=180); plt.close(fig)


def summarize(files, output):
    groups = {}
    for path in files:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        for dataset, result in report['results'].items():
            key = json.dumps([report['model_config'], report.get('protocol'), dataset, report['resolution'], report['manifest_sha256'], report['threshold']], sort_keys=True)
            group = groups.setdefault(key, {'model': report['model_config'], 'dataset': dataset,
                'resolution': report['resolution'], 'protocol': report.get('protocol'),
                'manifest_sha256': report['manifest_sha256'], 'threshold': report['threshold'], 'seeds': [], 'values': []})
            if report['seed'] in group['seeds']:
                raise ValueError('Duplicate seed report in one comparison group')
            group['seeds'].append(report['seed']); group['values'].append(result)
    summaries = []
    for group in groups.values():
        results = {}
        for aggregation in ['macro', 'global']:
            results[aggregation] = {}
            for metric in group['values'][0][aggregation]:
                numbers = [r[aggregation][metric] for r in group['values']]
                results[aggregation][metric] = {'mean': statistics.mean(numbers),
                    'sample_std': statistics.stdev(numbers) if len(numbers)>1 else None}
        summaries.append({k:v for k,v in group.items() if k != 'values'} | {'metrics': results})
    json_write(output, summaries)
    return summaries


def compare(checkpoints, output, kind='segmentation', ids=None, limit=4, root=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if kind not in {'segmentation','saliency','erf'}:
        raise ValueError(kind)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    first = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
    rows = sorted([r for r in read_manifest(first['config']['data']['manifest']) if r['split']=='test'],
                  key=lambda r:(r['dataset'],r['id']))
    if ids:
        rows = [r for r in rows if f"{r['dataset']}/{r['id']}" in ids]
        if len(rows) != len(set(ids)):
            raise ValueError('Requested IDs must match dataset/id in the test manifest')
    else:
        rows = rows[:limit]
    if not rows:
        raise ValueError('No comparison samples')
    columns, labels = [], []
    for path in checkpoints:
        model, ckpt = load_run(path, device)
        if ckpt['manifest_sha256'] != first['manifest_sha256']:
            raise ValueError('Comparison checkpoints must share one test manifest')
        ds = SegmentationDataset(rows, root or ckpt['config']['data']['root'],
                                  ckpt['config']['model']['image_size'], ckpt['stats'])
        values = []
        for item in ds:
            x = item['image'][None].to(device).requires_grad_(kind != 'segmentation')
            with torch.set_grad_enabled(kind != 'segmentation'):
                y = logits_of(model(x)).float()
                if kind == 'segmentation':
                    value = (y.sigmoid()[0,0] >= ckpt['config']['evaluation']['threshold']).float()
                else:
                    if kind == 'erf':
                        score = y[0,0,y.shape[-2]//2,y.shape[-1]//2]
                    else:
                        region = (y.detach().sigmoid()>=.5).float()
                        if not region.any():
                            region = torch.ones_like(region)
                        score = (y*region).sum()/region.sum()
                    grad, = torch.autograd.grad(score,x)
                    value = grad.abs().sum(1)[0]
            values.append(value.detach().cpu().numpy())
        columns.append(values)
        labels.append(f"{ckpt['config']['model']['name']}\n{Path(path).parent.name}")
        del model
    ds = SegmentationDataset(rows, root or first['config']['data']['root'],
                              first['config']['model']['image_size'], first['stats'])
    fig, axes = plt.subplots(len(rows),2+len(columns),figsize=(3*(2+len(columns)),3*len(rows)),squeeze=False)
    raw = {}
    for i,item in enumerate(ds):
        rgb = item['image']*torch.tensor(first['stats']['std'])[:,None,None]+torch.tensor(first['stats']['mean'])[:,None,None]
        axes[i,0].imshow(rgb.permute(1,2,0).clamp(0,1)); axes[i,1].imshow(item['mask'][0],cmap='gray',vmin=0,vmax=1)
        bound = max(float(column[i].max()) for column in columns) or 1.
        for j,column in enumerate(columns):
            value = column[i] if kind == 'segmentation' else np.sqrt(column[i]/bound)
            axes[i,j+2].imshow(value,cmap='gray' if kind=='segmentation' else 'Greens',vmin=0,vmax=1)
            raw[f'sample_{i}_model_{j}'] = column[i]
        for j,ax in enumerate(axes[i]):
            ax.axis('off')
            if i == 0:
                ax.set_title((['Image','Ground truth']+labels)[j],fontsize=8)
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=180); plt.close(fig)
    np.savez(output/'comparison_raw.npz',**raw)
    json_write(output/'selection.json',{'ids':[f"{r['dataset']}/{r['id']}" for r in rows],
                                      'checkpoints':[str(p) for p in checkpoints],'kind':kind,
                                      'normalization':'shared maximum across models for each sample; square root for gradients'})


def plot_comparison(summary, profiles, dataset, output, resolution='256'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries = json.loads(Path(summary).read_text(encoding='utf-8'))
    measurements = {}
    for path in profiles:
        p = json.loads(Path(path).read_text(encoding='utf-8'))
        measurements[json.dumps(p['model'],sort_keys=True)] = p
    points = []
    for s in summaries:
        if s['dataset'] != dataset or s['resolution'] != resolution:
            continue
        key = json.dumps(s['model'],sort_keys=True)
        if key not in measurements:
            raise ValueError(f"Missing matching profile for {s['model']}")
        points.append((s,measurements[key]))
    if not points:
        raise ValueError('No matching measured evaluation reports')
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for s,p in points:
        dsc = s['metrics']['macro']['dice']['mean']*100
        sd = s['metrics']['macro']['dice']['sample_std']
        for ax,x in zip(axes,[p['parameters']/1e6,p['estimated_counted_gflops']]):
            ax.errorbar(x,dsc,yerr=sd*100 if sd is not None else None,fmt='o')
            ax.annotate(s['model']['name'],(x,dsc),fontsize=7,xytext=(3,3),textcoords='offset points')
    for ax,label in zip(axes,['Parameters (M real scalars)','Counted GFLOPs estimate (partial)']):
        ax.set_xlabel(label); ax.set_ylabel('Macro DSC (%)'); ax.grid(alpha=.25)
    fig.suptitle(f'{dataset}, {resolution} evaluation'); fig.tight_layout()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180); plt.close(fig)
