"""Training and evaluation experiment matrix."""
from pathlib import Path
import subprocess
import sys
import yaml
from .config import load_config


def experiments(suite, config_root='configs', seeds=(42,43,44), execute=False, output='runs'):
    root = Path(config_root)
    suites = ['main', 'ablation'] if suite == 'all' else [suite]
    files = sorted(p for s in suites for p in (root/s).glob('*.yaml'))
    if not files:
        raise ValueError(f'No experiment configs in {root} for {suite}')
    jobs = []
    for path in files:
        for seed in seeds:
            run = Path(output)/path.parent.name/path.stem/f'seed{seed}'
            overrides = [f'train.seed={seed}', f'output={run.as_posix()}']
            config = load_config(path, overrides)
            cmd = [sys.executable, '-m', 'mamba_mis', 'train', '--config', str(path)]
            for override in overrides:
                cmd += ['--set', override]
            # Every run gets a separate final test report, outside the training loop.
            eval_cmd = [sys.executable, '-m', 'mamba_mis', 'evaluate', '--checkpoint', str(run/'best.pt'),
                        '--output', str(run/'test_256.json')]
            jobs.append({'config': str(path), 'seed': seed, 'output': str(run), 'train': cmd, 'evaluate': eval_cmd})
            if execute:
                subprocess.run(cmd, check=True)
                subprocess.run(eval_cmd, check=True)
    return jobs
