import argparse
import json
import torch.distributed as dist


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mamba-MIS medical image segmentation')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--dataset', choices=['isic17','isic18','polyp'], required=True)
    p.add_argument('--manifest'); p.add_argument('--groups'); p.add_argument('--seed', type=int, default=42)
    for name in ['train', 'profile']:
        p = sub.add_parser(name)
        p.add_argument('--config', required=True); p.add_argument('--set', action='append', default=[])
        if name == 'train':
            p.add_argument('--resume')
        else:
            p.add_argument('--output', required=True); p.add_argument('--warmup', type=int, default=5)
            p.add_argument('--repeats', type=int, default=20); p.add_argument('--device')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--root'); p.add_argument('--manifest'); p.add_argument('--original', action='store_true')
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint', required=True); p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('visualize')
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--kind', choices=['segmentation','erf','spectral','bridge','saliency'], default='segmentation')
    p.add_argument('--ids', nargs='+'); p.add_argument('--root'); p.add_argument('--limit', type=int, default=8)
    p = sub.add_parser('summarize')
    p.add_argument('--inputs', nargs='+', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('experiments')
    p.add_argument('--suite', choices=['main','ablation','all'], default='all')
    p.add_argument('--configs', default='configs'); p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--execute', action='store_true'); p.add_argument('--output', default='runs')
    p = sub.add_parser('models')
    p = sub.add_parser('compare')
    p.add_argument('--checkpoints', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--kind', choices=['segmentation','saliency','erf'], default='segmentation')
    p.add_argument('--ids', nargs='+'); p.add_argument('--limit', type=int, default=4); p.add_argument('--root')
    p = sub.add_parser('plot')
    p.add_argument('--summary', required=True); p.add_argument('--profiles', nargs='+', required=True)
    p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
    p.add_argument('--resolution', default='256')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            from .data import prepare
            result = prepare(args.root, args.output, args.dataset, args.seed, args.manifest, args.groups)
        elif args.command == 'train':
            from .engine import train
            from .config import load_config
            result = train(load_config(args.config, args.set), args.resume)
        elif args.command == 'evaluate':
            from .engine import evaluate
            result = evaluate(args.checkpoint, args.output, args.root, args.manifest, args.original)
        elif args.command == 'profile':
            from .analysis import profile
            from .config import load_config
            result = profile(load_config(args.config, args.set), args.output, args.warmup, args.repeats, args.device)
        elif args.command == 'predict':
            from .analysis import predict
            result = predict(args.checkpoint, args.inputs, args.output)
        elif args.command == 'visualize':
            from .analysis import visualize
            result = visualize(args.checkpoint, args.output, args.kind, args.ids, args.root, args.limit)
        elif args.command == 'summarize':
            from .analysis import summarize
            result = summarize(args.inputs, args.output)
        elif args.command == 'experiments':
            from .experiments import experiments
            result = experiments(args.suite, args.configs, args.seeds, args.execute, args.output)
        elif args.command == 'compare':
            from .analysis import compare
            result = compare(args.checkpoints,args.output,args.kind,args.ids,args.limit,args.root)
        elif args.command == 'plot':
            from .analysis import plot_comparison
            result = plot_comparison(args.summary,args.profiles,args.dataset,args.output,args.resolution)
        else:
            from .models import MODEL_NAMES
            result = list(MODEL_NAMES)
        if result is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
