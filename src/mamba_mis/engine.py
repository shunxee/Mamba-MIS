"""Train and evaluate with explicit split separation and epoch-boundary resumption."""
import contextlib
import copy
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import functional as F
from .config import json_write, sha256
from .data import (read_manifest, validate_manifest, compute_stats, SegmentationDataset,
                   EvaluationSampler)
from .models import build_model, logits_of
from .metrics import segmentation_loss, MetricAccumulator, confusion, from_confusion, METRICS


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def setup():
    world, rank, local = int(os.getenv('WORLD_SIZE', '1')), int(os.getenv('RANK', '0')), int(os.getenv('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, world


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def amp_context(device, mode):
    return torch.autocast(device.type, dtype=torch.bfloat16 if mode == 'bf16' else torch.float16,
                          enabled=device.type == 'cuda' and mode != 'no')


def load_pretrained(model, config):
    if not config:
        return None
    path = Path(config['path'])
    weights = torch.load(path, map_location='cpu', weights_only=True)
    for key in config.get('key', '').split('.'):
        if key:
            weights = weights[key]
    target = model.get_submodule(config['target']) if config.get('target') else model
    result = target.load_state_dict(weights, strict=config.get('strict', True))
    return {'path': str(path), 'sha256': sha256(path), 'target': config.get('target', ''),
            'missing': result.missing_keys, 'unexpected': result.unexpected_keys}


def runtime_metadata():
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'platform': platform.platform()}


def epoch_train(model, loader, optimizer, scaler, device, accumulation, settings):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros(2, dtype=torch.float64, device=device)
    micro = loader.batch_size
    local_samples = len(loader.sampler)
    for i, batch in enumerate(loader):
        group_start = (i // accumulation)*accumulation
        group_samples = min(accumulation*micro, local_samples-group_start*micro)
        boundary = (i+1) % accumulation == 0 or i+1 == len(loader)
        sync = contextlib.nullcontext() if boundary or not isinstance(model, DistributedDataParallel) else model.no_sync()
        images, masks = batch['image'].to(device), batch['mask'].to(device)
        with sync:
            with amp_context(device, settings['amp']):
                loss = segmentation_loss(model(images), masks, settings['aux_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at batch {i}')
            scaler.scale(loss*(images.shape[0]/group_samples)).backward()
        if boundary:
            scaler.unscale_(optimizer)
            if settings['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        total[0] += loss.detach().double()*images.shape[0]
        total[1] += images.shape[0]
    if dist.is_initialized():
        dist.all_reduce(total)
    return (total[0]/total[1]).item()


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=.5, amp='no', per_image=False):
    model = unwrap(model)
    model.eval()
    # Evaluate the same BN buffers on every rank without DDP's forward collectives.
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    metrics = MetricAccumulator(device)
    details = []
    for batch in loader:
        image, mask = batch['image'].to(device), batch['mask'].to(device)
        with amp_context(device, amp):
            logits = logits_of(model(image)).float()
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.sigmoid() >= threshold
        metrics.update(pred, mask >= .5)
        if per_image:
            values = from_confusion(confusion(pred, mask >= .5)).tolist()
            details.extend({'id': sample, **dict(zip(METRICS, val))} for sample, val in zip(batch['id'], values))
    metrics.synchronize()
    if per_image and dist.is_initialized():
        gathered = [None]*dist.get_world_size()
        dist.all_gather_object(gathered, details)
        details = [x for group in gathered for x in group]
    report = metrics.compute()
    if per_image:
        report['per_image'] = sorted(details, key=lambda v:v['id'])
    return report


def train(config, resume=None):
    device, rank, world = setup()
    settings = config['train']
    seed_all(settings['seed']+rank)
    torch.use_deterministic_algorithms(settings['deterministic'])
    micro, effective = settings['micro_batch_size'], settings['effective_batch_size']
    if effective % (micro*world):
        raise ValueError('effective_batch_size must be divisible by micro_batch_size * world_size')
    accumulation = effective // (micro*world)
    if accumulation < 1:
        raise ValueError('Effective batch size is too small')
    root, manifest = config['data']['root'], config['data']['manifest']
    rows = read_manifest(manifest)
    split_hash = sha256(manifest)
    if rank == 0:
        validate_manifest(rows, root)
    if dist.is_initialized():
        dist.barrier()
    stats = compute_stats(rows, root, config['model']['image_size']) if rank == 0 else None
    if dist.is_initialized():
        objects = [stats]; dist.broadcast_object_list(objects, src=0); stats = objects[0]
    training = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'val']
    if not training or not validation:
        raise ValueError('Both train and val splits are required')
    trainset = SegmentationDataset(training, root, config['model']['image_size'], stats, True, settings['seed'])
    valset = SegmentationDataset(validation, root, config['model']['image_size'], stats)
    sampler = DistributedSampler(trainset, world, rank, shuffle=True, seed=settings['seed'])
    trainloader = DataLoader(trainset, batch_size=micro, sampler=sampler, num_workers=settings['workers'],
                             pin_memory=device.type == 'cuda', persistent_workers=False)
    valloader = DataLoader(valset, batch_size=micro, sampler=EvaluationSampler(valset, rank, world),
                           num_workers=settings['workers'], pin_memory=device.type == 'cuda')
    model = build_model(config['model']).to(device)
    pretrained = load_pretrained(model, config.get('pretrained')) if not resume else None
    # Several upstream baseline networks retain unused classifier/deep supervision parameters.
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                         find_unused_parameters=not config['model']['name'].startswith('mamba_mis_'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['t_max'], eta_min=settings['eta_min'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and settings['amp'] == 'fp16')
    start, best = 0, -1.
    output = Path(config['output'])
    if resume:
        # Full local training checkpoints contain Python/NumPy RNG state; load trusted checkpoints only.
        ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        if ckpt['manifest_sha256'] != split_hash or ckpt['config']['model'] != config['model']:
            raise ValueError('Resume requires identical model and split manifest')
        if ckpt['config']['train'] != config['train']:
            raise ValueError('Resume requires the original training configuration')
        if len(ckpt['rng']) != world:
            raise ValueError('Exact resume requires the same world size')
        unwrap(model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler']); restore_rng(ckpt['rng'][rank])
        start, best, pretrained = ckpt['epoch']+1, ckpt['best'], ckpt['pretrained']
        if ckpt['stats'] != stats:
            raise ValueError('Training normalization changed since checkpoint')
    elif (output/'last.pt').exists():
        raise FileExistsError('Run exists; choose another output directory or use --resume')
    if rank == 0:
        json_write(output/'config.json', config)
        json_write(output/'normalization.json', stats)
    for epoch in range(start, settings['epochs']):
        sampler.set_epoch(epoch); trainset.epoch = epoch
        lr = optimizer.param_groups[0]['lr']
        loss = epoch_train(model, trainloader, optimizer, scaler, device, accumulation, settings)
        val = evaluate_loader(model, valloader, device, config['evaluation']['threshold'], settings['amp'])
        scheduler.step()
        improved = val['macro']['dice'] > best
        best = max(best, val['macro']['dice'])
        states = [None]*world
        if dist.is_initialized():
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            row = {'epoch': epoch+1, 'lr': lr, 'train_loss': loss, 'val': val}
            with (output/'history.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n')
            ckpt = {'model': unwrap(model).state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': states,
                    'epoch': epoch, 'best': best, 'config': copy.deepcopy(config), 'stats': stats,
                    'manifest_sha256': split_hash, 'pretrained': pretrained, 'runtime': runtime_metadata()}
            atomic_save(output/'last.pt', ckpt)
            if improved:
                atomic_save(output/'best.pt', ckpt)
            print(f"epoch={epoch+1} loss={loss:.5f} val_dice={val['macro']['dice']:.5f} lr={lr:.6g}", flush=True)
    if dist.is_initialized():
        dist.barrier()


def load_run(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['config']['model']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    return model.eval(), ckpt


def evaluate(checkpoint, output, root=None, manifest=None, original=False):
    device, rank, world = setup()
    model, ckpt = load_run(checkpoint, device)
    config = ckpt['config']
    root = root or config['data']['root']
    manifest = manifest or config['data']['manifest']
    rows = read_manifest(manifest)
    validate_manifest(rows, root)
    # Test membership may not be redefined after checkpoint selection.
    if sha256(manifest) != ckpt['manifest_sha256']:
        raise ValueError('Evaluation manifest must match the checkpoint manifest')
    test_rows = [r for r in rows if r['split'] == 'test']
    if not test_rows:
        raise ValueError('No test samples in manifest')
    results = {}
    for name in sorted({r['dataset'] for r in test_rows}):
        ds = SegmentationDataset([r for r in test_rows if r['dataset'] == name], root,
                                  config['model']['image_size'], ckpt['stats'], original=original)
        loader = DataLoader(ds, batch_size=1 if original else config['train']['micro_batch_size'],
                             sampler=EvaluationSampler(ds, rank, world), num_workers=config['train']['workers'])
        results[name] = evaluate_loader(model, loader, device, config['evaluation']['threshold'],
                                         config['train']['amp'], per_image=True)
    report = {'model': config['model']['name'], 'model_config': config['model'],
              'protocol': {'train': {k:v for k,v in config['train'].items() if k != 'seed'},
                           'pretrained_sha256': ckpt['pretrained']['sha256'] if ckpt['pretrained'] else None},
              'seed': config['train']['seed'], 'checkpoint_sha256': sha256(checkpoint),
              'manifest_sha256': ckpt['manifest_sha256'], 'resolution': 'original' if original else '256' if config['model']['image_size'] == 256 else str(config['model']['image_size']),
              'threshold': config['evaluation']['threshold'], 'results': results, 'runtime': runtime_metadata()}
    if rank == 0:
        json_write(output, report)
    return report
