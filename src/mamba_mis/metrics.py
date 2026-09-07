import torch
from torch.nn import functional as F


def dice_bce(logits, target):
    logits, target = logits.float(), target.float()
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    dice = 1 - (2*(p*t).sum(1)+1e-6)/(p.sum(1)+t.sum(1)+1e-6)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()


def segmentation_loss(output, target, aux_weight=0.0):
    if not isinstance(output, dict):
        return dice_bce(output, target)
    loss = dice_bce(output['logits'], target)
    aux = output.get('aux', [])
    if aux_weight and aux:
        loss = loss + aux_weight * torch.stack([dice_bce(o, target) for o in aux]).mean()
    return loss


def confusion(pred, target):
    pred, target = pred.bool().flatten(1), target.bool().flatten(1)
    return torch.stack([(pred & target).sum(1), (pred & ~target).sum(1),
                        (~pred & target).sum(1), (~pred & ~target).sum(1)], -1).double()


def from_confusion(c):
    tp, fp, fn, tn = c.unbind(-1)
    def divide(a, b, empty=1.):
        return torch.where(b > 0, a/b.clamp_min(1), torch.full_like(a, empty))
    fg, bg = divide(tp, tp+fp+fn), divide(tn, tn+fp+fn)
    return torch.stack([fg, (fg+bg)/2, divide(2*tp, 2*tp+fp+fn),
                        divide(tp+tn, tp+tn+fp+fn), divide(tn, tn+fp), divide(tp, tp+fn)], -1)


METRICS = ['foreground_iou', 'miou_fg_bg', 'dice', 'accuracy', 'specificity', 'sensitivity']


class MetricAccumulator:
    def __init__(self, device='cpu'):
        self.counts = torch.zeros(4, dtype=torch.float64, device=device)
        self.sums = torch.zeros(6, dtype=torch.float64, device=device)
        self.n = torch.zeros((), dtype=torch.float64, device=device)
    def update(self, pred, target):
        c = confusion(pred, target)
        self.counts += c.sum(0)
        self.sums += from_confusion(c).sum(0)
        self.n += c.shape[0]
    def synchronize(self):
        import torch.distributed as dist
        if dist.is_initialized():
            for v in [self.counts, self.sums, self.n]:
                dist.all_reduce(v)
    def compute(self):
        if self.n.item() == 0:
            raise ValueError('Cannot report metrics for an empty dataset')
        return {'samples': int(self.n.item()), 'confusion': self.counts.tolist(),
                'macro': dict(zip(METRICS, (self.sums/self.n).tolist())),
                'global': dict(zip(METRICS, from_confusion(self.counts).tolist()))}
