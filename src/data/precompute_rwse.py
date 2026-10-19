"""Build the RWSE caches, and check every split against committed data fingerprints.

Usage:
  python -m src.data.precompute_rwse                        # build caches, verify fingerprints
  python -m src.data.precompute_rwse --write-fingerprints   # record fingerprints (done once)

The fingerprint hashes node features, edges, edge features and labels of every
graph in order, plus the split size. A mismatch means the local dataset is not
byte-identical to the one the reimplementation was validated on, which is a
preregistered failure check (PREREGISTRATION.md 3.3).
"""
import argparse
import hashlib
import json
import os
import time

import torch

from src.data.datasets import RWSE_CACHE_STEPS, load_splits

FINGERPRINTS = 'configs/data_fingerprints.json'


def data_fingerprint(graphs):
    h = hashlib.sha256()
    for g in graphs:
        for t in (g.x, g.edge_index, g.edge_attr, g.y):
            h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--datasets', nargs='+', default=list(RWSE_CACHE_STEPS))
    parser.add_argument('--write-fingerprints', action='store_true')
    args = parser.parse_args()
    known = json.load(open(FINGERPRINTS)) if os.path.exists(FINGERPRINTS) else {}
    found, mismatches = {}, []
    for name in args.datasets:
        start = time.time()
        splits = load_splits(name, args.root, rwse_steps=RWSE_CACHE_STEPS[name])
        found[name] = {s: dict(graphs=len(gs), sha256=data_fingerprint(gs)) for s, gs in splits.items()}
        print(f'{name}: {({s: v["graphs"] for s, v in found[name].items()})}, '
              f'RWSE-{RWSE_CACHE_STEPS[name]} ready in {time.time() - start:.0f}s')
        sample = splits['train'][0].rwse[:2, :6]
        print(f'  first graph, nodes 0-1, RWSE steps 1-6: {torch.round(sample, decimals=3).tolist()}')
        if name in known and not args.write_fingerprints:
            for split, v in found[name].items():
                if known[name].get(split) != v:
                    mismatches.append(f'{name}/{split}')
    if args.write_fingerprints:
        known.update(found)
        with open(FINGERPRINTS, 'w') as f:
            json.dump(known, f, indent=2)
        print(f'wrote {FINGERPRINTS}')
    elif mismatches:
        raise SystemExit(f'DATA MISMATCH vs {FINGERPRINTS}: {", ".join(mismatches)}')
    elif known:
        print(f'all splits match {FINGERPRINTS}')


if __name__ == '__main__':
    main()
