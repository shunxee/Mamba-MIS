"""Local data only. CSV manifests make train/validation/test membership explicit."""
import csv
import hashlib
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler

POLYP_TESTS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
FIELDS = ['id', 'dataset', 'image', 'mask', 'split', 'group']
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def sample_key(path):
    key = Path(path).stem
    for suffix in ['_segmentation', '_mask', '_label']:
        if key.lower().endswith(suffix):
            key = key[:-len(suffix)]
    return key


def pair_folder(folder, dataset, split, root):
    folder, root = Path(folder), Path(root).resolve()
    image_dir, mask_dir = folder / 'images', folder / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError(f"Expected images/ and masks/ under {folder}")
    def index(directory):
        out = {}
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() in EXTENSIONS:
                key = sample_key(path)
                if key in out:
                    raise ValueError(f"Ambiguous duplicate sample ID {key} in {directory}")
                out[key] = path.resolve()
        return out
    images, masks = index(image_dir), index(mask_dir)
    if images.keys() != masks.keys() or not images:
        raise ValueError(f"Image/mask mismatch: missing masks={sorted(images.keys()-masks.keys())[:10]}, missing images={sorted(masks.keys()-images.keys())[:10]}")
    return [dict(id=k, dataset=dataset, image=images[k].relative_to(root).as_posix(),
                 mask=masks[k].relative_to(root).as_posix(), split=split, group='') for k in images]


def holdout(rows, count, seed, label='val', exact=False):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row.get('group') or row['id'])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    chosen, n = set(), 0
    for key in keys:
        size = len(groups[key])
        if n >= count:
            break
        if exact and n + size > count:
            continue
        chosen.add(key)
        n += size
    if exact and n != count:
        raise ValueError("Cannot obtain the exact split without dividing a group; supply an explicit manifest")
    result = []
    for row in rows:
        row = dict(row)
        key = (row['dataset'], row.get('group') or row['id'])
        row['split'] = label if key in chosen else 'train'
        result.append(row)
    return result


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not set(FIELDS[:-1]).issubset(rows[0]):
        raise ValueError(f"Manifest needs columns {FIELDS}; no usable rows in {path}")
    return rows


def validate_manifest(rows, root):
    root = Path(root)
    ids, paths, hashes, groups = set(), set(), {}, {}
    counts = defaultdict(int)
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid split {row['split']}")
        key = (row['dataset'], row['id'])
        if key in ids:
            raise ValueError(f"Duplicate sample identity {key}")
        ids.add(key)
        image_path, mask_path = root / row['image'], root / row['mask']
        if str(image_path.resolve()) in paths:
            raise ValueError(f"Repeated image path: {image_path}")
        paths.add(str(image_path.resolve()))
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Image/mask dimensions differ: {key}")
            pixels = np.asarray(image.convert('RGB'))
            digest = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
        if digest in hashes and hashes[digest] != row['split']:
            raise ValueError(f"Identical image appears across splits: {key}")
        hashes[digest] = row['split']
        if row.get('group'):
            g = (row['dataset'], row['group'])
            if g in groups and groups[g] != row['split']:
                raise ValueError(f"Group appears across splits: {g}")
            groups[g] = row['split']
        counts[f"{row['dataset']}/{row['split']}"] += 1
    return dict(counts)


def prepare(root, output, dataset, seed=42, source_manifest=None, groups_file=None):
    root = Path(root).resolve()
    if source_manifest:
        rows = read_manifest(source_manifest)
    elif dataset in {'isic17', 'isic18'}:
        label = dataset.upper()
        if (root / 'train').is_dir() and (root / 'test').is_dir():
            rows = pair_folder(root/'train', label, 'train', root) + pair_folder(root/'test', label, 'test', root)
            if (root / 'val').is_dir():
                rows += pair_folder(root/'val', label, 'val', root)
        else:
            rows = pair_folder(root, label, 'train', root)
            total, ntest = (2150, 650) if dataset == 'isic17' else (2694, 808)
            if len(rows) != total:
                raise ValueError(f"Expected {total} images; got {len(rows)}. Supply --manifest with explicit splits.")
            # Apply group IDs before either train/test or train/validation splitting.
            if groups_file:
                rows = attach_groups(rows, groups_file)
            rows = holdout(rows, ntest, seed, 'test', exact=True)
    elif dataset == 'polyp':
        rows = []
        if (root/'TrainDataset').is_dir():
            rows += pair_folder(root/'TrainDataset', 'polyp_train', 'train', root)
            for name in POLYP_TESTS:
                rows += pair_folder(root/'TestDataset'/name, name, 'test', root)
        else:
            for name in POLYP_TESTS:
                if name in POLYP_TESTS[:2]:
                    rows += pair_folder(root/name/'train', name, 'train', root)
                rows += pair_folder(root/name/'test', name, 'test', root)
    else:
        raise ValueError(dataset)
    if groups_file:
        rows = attach_groups(rows, groups_file)
    if not any(r['split'] == 'val' for r in rows):
        train = [r for r in rows if r['split'] == 'train']
        if len(train) < 2:
            raise ValueError("Need at least two training samples/groups to reserve validation")
        rows = [r for r in rows if r['split'] != 'train'] + holdout(train, max(1, round(.1*len(train))), seed)
    if not all(any(r['split'] == label for r in rows) for label in ['train', 'val', 'test']):
        raise ValueError('Preparation requires nonempty train, val, and test splits; check group sizes')
    counts = validate_manifest(rows, root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite split manifest: {output}")
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, '') for k in FIELDS} for r in sorted(rows, key=lambda r:(r['dataset'], r['id'])))
    output.with_suffix('.json').write_text(json.dumps({'seed': seed, 'counts': counts,
        'manifest_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    return counts


def attach_groups(rows, path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        groups = {(r['dataset'], r['id']): r['group'] for r in csv.DictReader(f)}
    return [dict(r, group=groups.get((r['dataset'], r['id']), r.get('group', ''))) for r in rows]


def compute_stats(rows, root, size=256):
    total, squared, count = np.zeros(3), np.zeros(3), 0
    for row in rows:
        if row['split'] != 'train':
            continue
        with Image.open(Path(root)/row['image']) as im:
            a = np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64)/255
        total += a.sum((0, 1)); squared += (a*a).sum((0, 1)); count += a.shape[0]*a.shape[1]
    if not count:
        raise ValueError('No training samples for normalization')
    mean = total/count
    std = np.sqrt(np.maximum(squared/count-mean*mean, 1e-8))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'pixels': count}


class SegmentationDataset(Dataset):
    def __init__(self, rows, root, size, stats, augment=False, seed=42, original=False):
        self.rows, self.root, self.size, self.stats = rows, Path(root), size, stats
        self.augment, self.seed, self.original, self.epoch = augment, seed, original, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['image']) as im:
            image = im.convert('RGB')
        with Image.open(self.root/row['mask']) as im:
            raw = np.asarray(im.convert('L'))
        mask = Image.fromarray(((raw > (0 if raw.max() <= 1 else 127))*255).astype(np.uint8))
        original_size = image.size[::-1]
        original_mask = torch.from_numpy(np.array(mask, copy=True)).float()[None]/255
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            rng = random.Random(self.seed + self.epoch*1000003 + index)
            if rng.random() < .5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if rng.random() < .5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            if rng.random() < .5:
                angle = rng.uniform(0, 360)
                image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=0)
                mask = mask.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()/255
        image = (image - torch.tensor(self.stats['mean'])[:, None, None])/torch.tensor(self.stats['std'])[:, None, None]
        mask = torch.from_numpy(np.array(mask, copy=True))[None].float()/255
        return {'image': image, 'mask': original_mask if self.original else mask,
                'id': row['id'], 'dataset': row['dataset'], 'original_size': original_size}


class EvaluationSampler(Sampler):
    """Unlike DistributedSampler, never pads or duplicates evaluation samples."""
    def __init__(self, dataset, rank=0, world_size=1):
        self.indices = list(range(rank, len(dataset), world_size))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
