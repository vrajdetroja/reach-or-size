import math

import torch

from src.data.datasets import nested_subset
from src.data.rwse import rwse


def _cycle(n):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def _closed_walks_on_cycle(n, k):
    # Walks of k +-1 steps whose sum is a multiple of n.
    return sum(math.comb(k, (k + j * n) // 2)
               for j in range(-k // n - 1, k // n + 2)
               if (k + j * n) % 2 == 0 and 0 <= (k + j * n) // 2 <= k)


def test_rwse_matches_closed_walk_counts_on_cycles():
    for n in (5, 6, 7):
        enc = rwse(_cycle(n), n, 16)
        for k in range(1, 17):
            expected = _closed_walks_on_cycle(n, k) / 2 ** k
            assert torch.allclose(enc[:, k - 1], torch.full((n,), expected), atol=1e-6)


def test_ring_is_invisible_to_short_walks():
    # On a cycle of length c, return probabilities match an infinite path until step c.
    big = rwse(_cycle(40), 40, 12)
    for c in (6, 8, 10):
        small = rwse(_cycle(c), c, 12)
        assert torch.allclose(small[0, :c - 1], big[0, :c - 1], atol=1e-6)
        assert small[0, c - 1] > big[0, c - 1]


def test_isolated_node_and_single_edge():
    enc = rwse(torch.tensor([[0, 1], [1, 0]]), 3, 4)
    assert torch.allclose(enc[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.all(enc[2] == 0)


def test_nested_subsets():
    a = set(nested_subset(1000, 0.25).tolist())
    b = set(nested_subset(1000, 0.5).tolist())
    assert len(a) == 250 and len(b) == 500 and a <= b
