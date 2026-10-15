import glob

import pytest
import torch
import yaml
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.data import Batch, Data

from src.data.datasets import TASKS
from src.gnnplus.layers import ModelConfig
from src.gnnplus.model import GNNPlus, count_parameters
from src.param_count import analytic_count
from src.train import build_model, rwse_steps

CONFIGS = sorted(glob.glob('configs/runs/*.yaml'))


def _random_batch(num_graphs, pe_steps, num_tasks, seed=0):
    g = torch.Generator().manual_seed(seed)
    atom_dims, bond_dims = get_atom_feature_dims(), get_bond_feature_dims()
    graphs = []
    for _ in range(num_graphs):
        n = int(torch.randint(5, 12, (1,), generator=g))
        src = torch.arange(n - 1)
        edge_index = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])
        x = torch.stack([torch.randint(0, d, (n,), generator=g) for d in atom_dims], dim=1)
        e = torch.stack([torch.randint(0, d, (n - 1,), generator=g) for d in bond_dims], dim=1)
        data = Data(x=x, edge_index=edge_index, edge_attr=torch.cat([e, e]), y=torch.rand(1, num_tasks, generator=g))
        if pe_steps:
            data.rwse = torch.rand(n, pe_steps, generator=g)
        graphs.append(data)
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize('path', CONFIGS)
def test_config_builds_with_reported_parameter_count(path):
    cfg = yaml.safe_load(open(path))
    task = TASKS[cfg['dataset']]
    model = build_model(cfg, task['num_tasks'])
    if 'expected_params' in cfg:
        assert count_parameters(model) == cfg['expected_params']
    batch = _random_batch(4, rwse_steps(cfg), task['num_tasks'])
    model.train()
    out = model(batch)
    assert out.shape == (4, task['num_tasks'])
    out.sum().backward()


@pytest.mark.parametrize('edge_encoder', ['shared', 'shared_pre_linear', 'shared_linear', 'per_layer_bond'])
@pytest.mark.parametrize('conv', ['gcn', 'gatedgcn'])
def test_readings_run_and_counts_agree(edge_encoder, conv):
    cfg = ModelConfig(conv=conv, hidden=32, layers=2, edge_encoder=edge_encoder, pe_steps=8, pe_dim=4,
                      residual=True, ffn=True, head='halving', head_layers=3, num_tasks=3)
    model = GNNPlus(cfg)
    assert count_parameters(model) == analytic_count(cfg)
    model(_random_batch(3, 8, 3)).sum().backward()


def test_gcn_layer_matches_dense_formula_without_edges():
    # Eq. 2 on a path graph: h = D^-1/2 (A + I) D^-1/2 X W + b.
    cfg = ModelConfig(hidden=8, edge=False)
    from src.gnnplus.layers import GCNPlusConv
    conv = GCNPlusConv(cfg)
    n = 5
    src = torch.arange(n - 1)
    edge_index = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])
    a = torch.eye(n)
    a[edge_index[0], edge_index[1]] = 1
    d = a.sum(1).pow(-0.5)
    x = torch.randn(n, 8)
    expected = (d[:, None] * a * d[None, :]) @ conv.lin(x) + conv.bias
    assert torch.allclose(conv(x, edge_index, None, None), expected, atol=1e-5)
