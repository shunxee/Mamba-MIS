import csv
import numpy as np
import pytest
from PIL import Image
import torch
from mamba_mis.data import (validate_manifest, holdout, compute_stats, SegmentationDataset,
                             EvaluationSampler, prepare, read_manifest)
from mamba_mis.metrics import MetricAccumulator, confusion, from_confusion, dice_bce


def fixture_rows(tmp_path, counts=(4, 2, 2)):
    rows = []
    rng = np.random.default_rng(77)
    for split, count in zip(['train','val','test'], counts):
        for i in range(count):
            name = f'{split}{i}'
            image = rng.integers(0, 255, (32,32,3), dtype=np.uint8)
            mask = (image[...,0] > 128).astype(np.uint8)*255
            Image.fromarray(image).save(tmp_path/f'{name}.png')
            Image.fromarray(mask).save(tmp_path/f'{name}_mask.png')
            rows.append(dict(id=name, dataset='synthetic', image=f'{name}.png',
                             mask=f'{name}_mask.png', split=split, group=name))
    path = tmp_path/'manifest.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows, path


def test_confusion_hand_computed_and_aggregation():
    p = torch.tensor([[[[1,1],[0,0]]], [[[1,0],[0,0]]]])
    t = torch.tensor([[[[1,0],[1,0]]], [[[0,0],[0,0]]]])
    expected = torch.tensor([1/3,1/3,.5,.5,.5,.5], dtype=torch.float64)
    torch.testing.assert_close(from_confusion(confusion(p[:1],t[:1]))[0], expected)
    one, batches = MetricAccumulator(), MetricAccumulator()
    one.update(p,t)
    for i in range(2):
        batches.update(p[i:i+1],t[i:i+1])
    assert one.compute() == batches.compute()
    assert one.compute()['macro']['dice'] != one.compute()['global']['dice']


def test_empty_masks_and_extreme_logits():
    zero = torch.zeros(1,1,2,2)
    torch.testing.assert_close(from_confusion(confusion(zero,zero)), torch.ones(1,6,dtype=torch.float64))
    for logit in [-1000.,1000.]:
        assert torch.isfinite(dice_bce(torch.full_like(zero,logit),zero))
    with pytest.raises(ValueError):
        MetricAccumulator().compute()


def test_split_and_train_only_stats(tmp_path):
    rows, _ = fixture_rows(tmp_path)
    validate_manifest(rows, tmp_path)
    stats = compute_stats(rows, tmp_path, 32)
    for r in rows:
        if r['split'] != 'train':
            Image.new('RGB',(32,32),(255,255,255)).save(tmp_path/r['image'])
    assert compute_stats(rows, tmp_path, 32) == stats
    rows[-1]['image'] = rows[0]['image']
    with pytest.raises(ValueError):
        validate_manifest(rows,tmp_path)


def test_group_split_deterministic_and_sampler():
    rows = [dict(id=str(i),dataset='d',group=str(i//2),split='train') for i in range(20)]
    a, b = holdout(rows,4,42), holdout(rows,4,42)
    assert a == b
    assert len({r['group'] for r in a if r['split']=='val'}) == 2
    shards = [list(EvaluationSampler(range(11),r,4)) for r in range(4)]
    assert sorted(x for shard in shards for x in shard) == list(range(11))


def test_augmentation_is_paired_binary_and_epoch_deterministic(tmp_path):
    # Exact correspondence survives flips; interpolation boundaries may differ during rotation.
    rows, _ = fixture_rows(tmp_path)
    ds = SegmentationDataset(rows,tmp_path,32,dict(mean=[0,0,0],std=[1,1,1]),True,42)
    a, b = ds[0], ds[0]
    assert torch.equal(a['image'],b['image']) and torch.equal(a['mask'],b['mask'])
    assert set(a['mask'].unique().tolist()) <= {0.,1.}
    agreement = ((a['image'][:1]>.5) == a['mask'].bool()).float().mean()
    assert agreement > .80
    ds.epoch = 8
    assert not torch.equal(ds[0]['image'],a['image'])


def test_prepare_explicit_manifest(tmp_path):
    rows, path = fixture_rows(tmp_path)
    output = tmp_path/'prepared.csv'
    prepare(tmp_path,output,'isic17',source_manifest=path)
    assert len(read_manifest(output)) == len(rows)
    with pytest.raises(FileExistsError):
        prepare(tmp_path,output,'isic17',source_manifest=path)
