import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train import cosine_with_warmup, metric_fn


def test_ap_is_mean_over_tasks_with_both_classes():
    y = torch.tensor([[1., 0., 1.], [0., 0., 1.], [1., 0., 1.], [0., 0., 1.]])
    pred = torch.tensor([[2., 1., 0.], [-1., 0., 1.], [0.5, 2., 3.], [0., 1., 2.]])
    # Tasks 1 (all zeros) and 2 (all ones) are skipped, as in LRGB.
    assert math.isclose(metric_fn('ap', pred, y), average_precision_score(y[:, 0], pred[:, 0]))


def test_mae_and_rocauc():
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    pred = torch.tensor([[0.1], [0.4], [0.9], [-0.3]])
    assert math.isclose(metric_fn('mae', pred, y), float(np.abs(pred - y).mean()), rel_tol=1e-6)
    assert math.isclose(metric_fn('rocauc', pred, y), roc_auc_score(y.flatten(), pred.flatten()))
    assert metric_fn('acc', pred, y) == 0.75


def test_cosine_with_warmup_shape():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_with_warmup(opt, warmup=5, total=105)
    lrs = []
    for _ in range(106):
        lrs.append(opt.param_groups[0]['lr'])
        sched.step()
    assert lrs[0] == 1e-6 and math.isclose(lrs[1], 0.2) and math.isclose(lrs[5], 1.0)
    assert math.isclose(lrs[55], 0.5, abs_tol=1e-9) and lrs[105] < 1e-9


def test_precise_bn_replaces_running_stats_with_training_set_average():
    from src.train import recompute_bn_stats

    class Batch:
        def __init__(self, x):
            self.x = x

        def to(self, device):
            return self

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)

        def forward(self, b):
            return self.bn(b.x)

    net = Net()
    batches = [Batch(torch.tensor([[1., 10.], [3., 10.]])), Batch(torch.tensor([[5., 20.], [7., 20.]]))]
    recompute_bn_stats(net, batches, 'cpu')
    assert torch.allclose(net.bn.running_mean, torch.tensor([4., 15.]))  # cumulative average of batch means
    assert net.bn.momentum == 0.1  # restored
