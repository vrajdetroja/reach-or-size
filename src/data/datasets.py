"""Loaders for the three in-scope datasets, with standard splits (LRGB, OGB)."""
import torch

from src.data.rwse import attach_rwse

# Metric, loss and output size per dataset (AMBIGUITIES.md A15: standard protocols).
TASKS = {
    'peptides-func': dict(metric='ap', loss='bce', num_tasks=10, higher_is_better=True),
    'peptides-struct': dict(metric='mae', loss='l1', num_tasks=11, higher_is_better=False),
    'ogbg-molhiv': dict(metric='rocauc', loss='bce', num_tasks=1, higher_is_better=True),
}

# Longest walk any config needs; shorter RWSE-K are prefixes of this.
RWSE_CACHE_STEPS = {'peptides-func': 32, 'peptides-struct': 32, 'ogbg-molhiv': 20}


def task_info(name):
    if name.startswith('ring-'):  # synthetic ring-size task (src/extension/synthetic.py)
        return dict(metric='acc', loss='bce', num_tasks=1, higher_is_better=True)
    return TASKS[name]


def _raw_splits(name, root):
    if name.startswith('ring-'):
        from src.extension.synthetic import parse_name, ring_splits
        return ring_splits(*parse_name(name))
    if name.startswith('peptides'):
        from torch_geometric.datasets import LRGBDataset
        return {s: LRGBDataset(f'{root}/lrgb', name=name, split=s) for s in ('train', 'val', 'test')}
    if name == 'ogbg-molhiv':
        from ogb.graphproppred import PygGraphPropPredDataset
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        # ogb 1.3.6 calls torch.load without weights_only; allow-list only PyG's data classes
        # (the processed file is built locally from OGB's CSVs) instead of disabling safe loading.
        torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        ds = PygGraphPropPredDataset(name, root=f'{root}/ogb')
        idx = ds.get_idx_split()
        return {'train': ds[idx['train']], 'val': ds[idx['valid']], 'test': ds[idx['test']]}
    raise ValueError(f'unknown dataset {name}')


def load_splits(name, root='data', rwse_steps=0):
    """Return {'train', 'val', 'test'} lists of Data objects.

    Labels become float [1, num_tasks]. With rwse_steps > 0 every graph gets
    `rwse` of shape [num_nodes, rwse_steps], read from the cache in root/rwse.
    """
    splits = {}
    for split, ds in _raw_splits(name, root).items():
        graphs = [g for g in ds]
        for g in graphs:
            g.y = g.y.view(1, -1).float()
        if rwse_steps:
            cache_steps = max(rwse_steps, RWSE_CACHE_STEPS.get(name, 32))
            attach_rwse(graphs, cache_steps, f'{root}/rwse/{name}_{split}_k{cache_steps}.pt')
            for g in graphs:
                g.rwse = g.rwse[:, :rwse_steps]
        splits[split] = graphs
    return splits


def nested_subset(num_graphs, fraction, subset_seed=0):
    """Indices of a random training subset; fractions with one seed are nested.

    Arm A (PREREGISTRATION.md 5.3) uses one fixed subset_seed for all model seeds.
    """
    if fraction >= 1.0:
        return torch.arange(num_graphs)
    perm = torch.randperm(num_graphs, generator=torch.Generator().manual_seed(subset_seed))
    return perm[:round(fraction * num_graphs)].sort().values
