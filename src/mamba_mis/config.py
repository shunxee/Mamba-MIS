from pathlib import Path
import copy
import hashlib
import json
import yaml


DEFAULT = {
    'model': {'name': 'mamba_mis_b', 'image_size': 256, 'backend': 'auto', 'use_checkpoint': True},
    'data': {'root': 'data/ISIC17', 'manifest': 'data/isic17.csv'},
    'train': {'epochs': 300, 'effective_batch_size': 32, 'micro_batch_size': 4,
              'lr': .001, 'weight_decay': .01, 't_max': 50, 'eta_min': .00001,
              'amp': 'bf16', 'workers': 4, 'aux_weight': 0., 'seed': 42,
              'deterministic': False, 'grad_clip': 1.0},
    'evaluation': {'threshold': .5}, 'output': 'runs/isic17/mamba_mis_b/seed42',
    'pretrained': None,
}


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path, overrides=()):
    def read(path, seen):
        path = Path(path).resolve()
        if path in seen:
            raise ValueError('Configuration inheritance cycle')
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        parent = data.pop('extends', None)
        return merge(read(path.parent/parent, seen | {path}), data) if parent else data
    config = merge(DEFAULT, read(path, set()))
    for expression in overrides:
        key, value = expression.split('=', 1)
        target = config
        parts = key.split('.')
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = yaml.safe_load(value)
    if config['train']['amp'] is False:
        config['train']['amp'] = 'no'
    if config['train']['amp'] not in {'no', 'bf16', 'fp16'}:
        raise ValueError('amp must be no, bf16, or fp16')
    if config['train']['deterministic']:
        config['model']['backend'] = 'torch'
    return config


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
