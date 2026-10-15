"""Build the RWSE caches on CPU, once, before training.

Usage: python -m src.data.precompute_rwse [--root data]
"""
import argparse
import time

import torch

from src.data.datasets import RWSE_CACHE_STEPS, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--datasets', nargs='+', default=list(RWSE_CACHE_STEPS))
    args = parser.parse_args()
    for name in args.datasets:
        start = time.time()
        splits = load_splits(name, args.root, rwse_steps=RWSE_CACHE_STEPS[name])
        sizes = {s: len(g) for s, g in splits.items()}
        nodes = sum(g.num_nodes for gs in splits.values() for g in gs)
        sample = splits['train'][0].rwse[:3, :8]
        print(f'{name}: {sizes}, {nodes} nodes, RWSE-{RWSE_CACHE_STEPS[name]} '
              f'in {time.time() - start:.0f}s')
        print(f'  first graph, first 3 nodes, steps 1-8:\n{torch.round(sample, decimals=3)}')


if __name__ == '__main__':
    main()
