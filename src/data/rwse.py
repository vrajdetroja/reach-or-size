"""Random-walk structural encoding (RWSE), the PE of Eq. 12.

RWSE_k(v) is the probability that a simple random walk started at v is back at
v after k steps: the v-th diagonal entry of P^k, with P = D^{-1} A. We store
steps 1..K (AMBIGUITIES.md A7, blind default). RWSE-K for a smaller K is the
first K columns, so one RWSE-32 cache serves arm C's K in {6, 16, 32}.

A closed walk of length k never goes further than floor(k/2) hops from its
start, so RWSE-K carries structure from at most floor(K/2) hops away.
"""
import hashlib
import os

import torch


def rwse(edge_index, num_nodes, steps):
    """Return-probability matrix of shape [num_nodes, steps] for one graph."""
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(dim=1)
    inv_deg = torch.where(deg > 0, 1.0 / deg.clamp(min=1.0), torch.zeros_like(deg))
    transition = inv_deg[:, None] * adj
    out = torch.empty(num_nodes, steps, dtype=torch.float64)
    power = transition
    for k in range(steps):
        out[:, k] = power.diagonal()
        power = power @ transition
    return out.float()


def _fingerprint(graphs):
    h = hashlib.sha256()
    for g in graphs:
        h.update(int(g.num_nodes).to_bytes(4, 'little'))
        h.update(g.edge_index.numpy().tobytes())
    return h.hexdigest()[:16]


def attach_rwse(graphs, steps, cache_path):
    """Set `g.rwse` ([num_nodes, steps]) on every graph, using a disk cache.

    The cache stores the concatenated encodings plus a fingerprint of the graph
    structures, so a cache built for a different graph order fails loudly.
    """
    fingerprint = _fingerprint(graphs)
    if os.path.exists(cache_path):
        cached = torch.load(cache_path)
        if cached['fingerprint'] != fingerprint or cached['steps'] < steps:
            raise ValueError(f'RWSE cache {cache_path} does not match these graphs')
        table = cached['rwse'][:, :steps]
    else:
        table = torch.cat([rwse(g.edge_index, g.num_nodes, steps) for g in graphs])
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({'fingerprint': fingerprint, 'steps': steps, 'rwse': table}, cache_path)
    offset = 0
    for g in graphs:
        g.rwse = table[offset:offset + g.num_nodes]
        offset += g.num_nodes
    if offset != table.size(0):
        raise ValueError(f'RWSE cache {cache_path} has {table.size(0)} rows, graphs have {offset}')
    return graphs
