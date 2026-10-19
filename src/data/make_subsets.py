"""Write arm A's nested training subsets as committed index files (PREREGISTRATION.md 5.3).

Usage: python -m src.data.make_subsets
Writes configs/subsets/peptides-func_train_frac{25,50}_seed0.json. The trainer reads
these through train.subset_file and checks the stored hash, so every machine trains
on exactly the same graphs.
"""
import hashlib
import json
import os

from src.data.datasets import nested_subset

NUM_TRAIN = 10873  # Peptides-func training split (LRGB)
SUBSET_SEED = 0


def main():
    os.makedirs('configs/subsets', exist_ok=True)
    previous = None
    for pct in (25, 50):
        idx = nested_subset(NUM_TRAIN, pct / 100, SUBSET_SEED).tolist()
        if previous is not None:
            assert set(previous) <= set(idx), 'subsets must be nested'
        previous = idx
        spec = dict(dataset='peptides-func', split='train', num_train_graphs=NUM_TRAIN, fraction=pct / 100,
                    subset_seed=SUBSET_SEED, size=len(idx),
                    sha256=hashlib.sha256(','.join(map(str, idx)).encode()).hexdigest(), indices=idx)
        path = f'configs/subsets/peptides-func_train_frac{pct}_seed{SUBSET_SEED}.json'
        with open(path, 'w') as f:
            json.dump(spec, f)
        print(f'{path}: {len(idx)} graphs, sha256 {spec["sha256"][:16]}')


if __name__ == '__main__':
    main()
