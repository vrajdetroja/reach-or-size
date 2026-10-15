"""Run the synthetic ring-size grid (PREREGISTRATION.md 5.7) on CPU.

Grid: ring pairs x depth L in {1, 2, 3, 4, 6, 8} x RWSE-K in {none, 4, 8, 16, 32} x seeds {0, 1, 2}.
Settings the preregistration leaves open, fixed here before any grid run:
hidden 64, 2-layer MLP head, RWSE width 16 (GraphGPS-style encoder), dropout 0,
AdamW lr 1e-3, batch 32, 60 epochs with 5 warmup epochs, test at best validation.

Usage: python -m src.extension.run_synthetic [--pairs 4-6 8-10] [--depths 1 2] [--ks 0 8] [--seeds 0]
Results: results/synthetic/<run>/ and one row per run in results/synthetic/summary.csv.
"""
import argparse
import csv
import os

from src.extension.synthetic import PAIRS
from src.train import run

DEPTHS = [1, 2, 3, 4, 6, 8]
KS = [0, 4, 8, 16, 32]


def config(pair, depth, k):
    return {
        'dataset': f'ring-{pair[0]}-{pair[1]}',
        'arch': 'gnnplus',
        'model': dict(conv='gcn', layers=depth, hidden=64, edge=False, bond_encoder_always=False, bn=True,
                      dropout=0.0, residual=True, ffn=False, pe_steps=k, pe_dim=16, pooling='mean',
                      head='mlp', head_layers=2),
        'train': dict(batch_size=32, lr=0.001, weight_decay=0.0, epochs=60, warmup_epochs=5,
                      scheduler='cosine_with_warmup', clip_grad_norm=0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=[f'{a}-{b}' for a, b in PAIRS])
    parser.add_argument('--depths', nargs='+', type=int, default=DEPTHS)
    parser.add_argument('--ks', nargs='+', type=int, default=KS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--out', default='results/synthetic')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, 'summary.csv')
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            done = {r['run'] for r in csv.DictReader(f)}
    fields = ['run', 'pair', 'depth', 'k', 'seed', 'best_epoch', 'best_val', 'test_at_best_val', 'params']
    new_file = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for pair in args.pairs:
            a, b = (int(x) for x in pair.split('-'))
            for depth in args.depths:
                for k in args.ks:
                    for seed in args.seeds:
                        name = f'ring-{a}-{b}_L{depth}_K{k}_s{seed}'
                        if name in done:
                            continue
                        s = run(config((a, b), depth, k), seed, args.out, name, epochs=args.epochs, device='cpu')
                        writer.writerow(dict(run=name, pair=pair, depth=depth, k=k, seed=seed,
                                             best_epoch=s['best_epoch'], best_val=s['best_val'],
                                             test_at_best_val=s['test_at_best_val'], params=s['params']))
                        f.flush()


if __name__ == '__main__':
    main()
