"""Two CPU/Gloo processes check collective aggregation, including unequal shards."""
from pathlib import Path
import json
import platform
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mamba_mis.data import EvaluationSampler
from mamba_mis.metrics import MetricAccumulator


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        acc = MetricAccumulator()
        for index in EvaluationSampler(range(7),rank,2):
            pred = torch.tensor([[[[index%2,1],[0,1]]]])
            target = torch.tensor([[[[1,1],[0,index%2]]]])
            acc.update(pred,target)
        acc.synchronize()
        if rank == 0:
            Path(output).write_text(json.dumps(acc.compute()),encoding='utf-8')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(platform.system() != 'Linux', reason='Linux Gloo validation; this Windows build has no usable Gloo device')
def test_two_process_metric_reduction(tmp_path):
    output = tmp_path/'distributed.json'
    mp.spawn(worker,args=((tmp_path/'rendezvous').as_uri(),str(output)),nprocs=2,join=True)
    expected = MetricAccumulator()
    for i in range(7):
        expected.update(torch.tensor([[[[i%2,1],[0,1]]]]),torch.tensor([[[[1,1],[0,i%2]]]]))
    assert json.loads(output.read_text()) == expected.compute()
