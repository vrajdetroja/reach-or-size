import random

import torch

from src.data.rwse import rwse
from src.extension.synthetic import ring_graph, ring_splits


def test_ring_graph_structure():
    for c in (4, 5, 6, 16):
        g = ring_graph(c, 50, random.Random(c))
        assert g.num_nodes == 50
        # A path plus one ring has exactly one cycle: |E| = |V|.
        assert g.edge_index.size(1) == 2 * 50
        degrees = torch.bincount(g.edge_index[0], minlength=50)
        assert int((degrees == 3).sum()) <= 1 and int((degrees == 1).sum()) == 2


def test_ring_is_a_c_cycle_and_rwse_has_no_odd_returns_for_even_rings():
    for c in (6, 8):
        g = ring_graph(c, 50, random.Random(0))
        ring_nodes = set(range(g.num_nodes - (c - 1), g.num_nodes))
        attach = {int(u) for u, v in g.edge_index.t().tolist() if v in ring_nodes and u not in ring_nodes}
        assert len(attach) == 1
        cycle = ring_nodes | attach
        inside = [(u, v) for u, v in g.edge_index.t().tolist() if u in cycle and v in cycle]
        assert len(inside) == 2 * c
        # Path plus an even ring is bipartite, so odd-step return probabilities are zero.
        enc = rwse(g.edge_index, g.num_nodes, 9)
        assert torch.all(enc[:, 0::2] == 0)


def test_splits_balanced_and_size_independent():
    splits = ring_splits(4, 6, sizes=(200, 50, 50))
    ys = torch.cat([g.y for g in splits['train']]).flatten()
    assert ys.sum() == 100
    n0 = [g.num_nodes for g in splits['train'] if g.y.item() == 0]
    n1 = [g.num_nodes for g in splits['train'] if g.y.item() == 1]
    assert abs(sum(n0) / len(n0) - sum(n1) / len(n1)) < 5
