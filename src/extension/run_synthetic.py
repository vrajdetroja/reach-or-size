"""Run the synthetic ring-size grid (PREREGISTRATION.md 5.7) on CPU.

Grid: ring pairs x depth L in {1, 2, 3, 4, 6, 8} x RWSE-K in {none, 4, 8, 16, 32} x seeds {0, 1, 2}.
Settings the preregistration leaves open, fixed here before any grid run:
hidden 64, 2-layer MLP head, RWSE width 16 (GraphGPS-style encoder), dropout 0,
AdamW lr 1e-3, batch 32, 60 epochs with 5 warmup epochs, test at best validation.
Added 2026-10-21 before the grid (PREREGISTRATION.md 7): precise BN before every evaluation. With
constant node features, BatchNorm running averages lag the model and eval-mode outputs collapse to
one class while training accuracy is 1.0; recomputing the statistics on the training set fixes it.

Usage: python -m src.extension.run_synthetic [--pairs 4-6 8-10] [--depths 1 2] [--ks 0 8] [--seeds 0]
Results: results/synthetic/<run>/; a run whose summary.json exists is skipped, so parallel runs
share nothing. analysis/synthetic.py collects the per-run summaries.
"""
import argparse
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
                      scheduler='cosine_with_warmup', clip_grad_norm=0.0, precise_bn=True),
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
    for pair in args.pairs:
        a, b = (int(x) for x in pair.split('-'))
        for depth in args.depths:
            for k in args.ks:
                for seed in args.seeds:
                    name = f'ring-{a}-{b}_L{depth}_K{k}_s{seed}'
                    if os.path.exists(os.path.join(args.out, name, 'summary.json')):
                        continue
                    run(config((a, b), depth, k), seed, args.out, name, epochs=args.epochs, device='cpu')


if __name__ == '__main__':
    main()
