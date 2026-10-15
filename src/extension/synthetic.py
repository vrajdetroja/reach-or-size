"""Synthetic ring-size task (PREREGISTRATION.md 5.7), CPU only.

Each graph is a path with one ring attached at a random interior node; the ring
shares that node with the path. The label says which of two ring sizes was
attached. Total node count is drawn from U[n_min, n_max] independently of the
label, so graph size carries no signal. Node features are constant (all-zero
categorical features, so the atom encoder gives every node the same vector)
and there are no edge features.

Predictions being tested, for a pair (c, c + 2):
- H-reach mechanism: without RWSE, accuracy stays near chance for L < c/2 and
  rises once L >= c/2; with RWSE-K, accuracy is high whenever K >= c, at any L.
- H-local mechanism: without RWSE, accuracy stays near chance even at
  L >= c/2 + 2, i.e. GCN+ cannot compute ring size inside its receptive field.
"""
import random

import torch
from torch_geometric.data import Data

PAIRS = [(4, 6), (8, 10), (12, 14), (16, 18), (5, 6)]


def ring_graph(ring_size, total_nodes, rng):
    path_len = total_nodes - ring_size + 1
    attach = rng.randrange(1, path_len - 1)
    edges = [(i, i + 1) for i in range(path_len - 1)]
    ring = [attach] + list(range(path_len, path_len + ring_size - 1))
    edges += [(ring[i], ring[(i + 1) % ring_size]) for i in range(ring_size)]
    src, dst = zip(*edges)
    edge_index = torch.tensor([src + dst, dst + src])
    return Data(x=torch.zeros(total_nodes, 9, dtype=torch.long), edge_index=edge_index,
                edge_attr=torch.zeros(edge_index.size(1), 3, dtype=torch.long), num_nodes=total_nodes)


def ring_splits(small, large, sizes=(2000, 500, 500), n_range=(40, 70), seed=0):
    """Balanced binary splits: label 1 means the larger ring."""
    rng = random.Random(f'ring-{small}-{large}-{seed}')
    splits = {}
    for split, n in zip(('train', 'val', 'test'), sizes):
        graphs = []
        for i in range(n):
            label = i % 2
            g = ring_graph(large if label else small, rng.randint(*n_range), rng)
            g.y = torch.tensor([[float(label)]])
            graphs.append(g)
        rng.shuffle(graphs)
        splits[split] = graphs
    return splits


def parse_name(name):
    """'ring-4-6' -> (4, 6)."""
    _, small, large = name.split('-')
    return int(small), int(large)
