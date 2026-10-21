"""Discrepancy measurements (DISCREPANCIES.md), GCN+ on Peptides-func, 3 seeds each.

Usage: python -m analysis.discrepancy [--runs results/runs] [--disc results/discrepancy]

Baseline: the code-aligned C1 runs (results/runs/gcnplus_peptides-func_s*). Each row changes one
factor back to the paper's reading; the last row changes all of them (the paper-literal model).
A row counts as material if |effect| > 2 * SE, SE = sqrt(s_base^2/n_base + s_row^2/n_row).
"""
import argparse
import glob
import json
import math
import os
import statistics

ROWS = [
    ('1', 'Edge message as Eq. 6 is typeset', 'gcnplus_peptides-func_d1_edge_message_paper'),
    ('2', 'ReLU instead of GELU', 'gcnplus_peptides-func_d2_activation_relu'),
    ('3+4', 'No head dropout, no gradient clipping', 'gcnplus_peptides-func_d3_no_headdrop_noclip'),
    ('all', 'All of the above (paper-literal)', 'gcnplus_peptides-func'),
]


def scores(d, stem):
    return [json.load(open(f))['test_at_best_val'] for f in sorted(glob.glob(os.path.join(d, stem + '_s*', 'summary.json')))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='results/runs')
    parser.add_argument('--disc', default='results/discrepancy')
    args = parser.parse_args()
    base = scores(args.runs, 'gcnplus_peptides-func')
    mb, sb = statistics.mean(base), statistics.stdev(base)
    print('# Discrepancy measurements: GCN+ Peptides-func, test AP at best val\n')
    print(f'Baseline, code-aligned C1 (results/runs): {mb:.4f} ± {sb:.4f} ({len(base)} seeds). Paper: 0.7261 ± 0.0067.\n')
    print('| Row | Change from the code-aligned model | Test AP | Effect vs. baseline | 2·SE | Material? |')
    print('| --- | --- | --- | --- | --- | --- |')
    for row, text, stem in ROWS:
        xs = scores(args.disc, stem)
        if len(xs) < 2:
            print(f'| {row} | {text} | — ({len(xs)} seeds) | | | not run |')
            continue
        m, s = statistics.mean(xs), statistics.stdev(xs)
        effect = m - mb
        se2 = 2 * math.sqrt(sb ** 2 / len(base) + s ** 2 / len(xs))
        print(f'| {row} | {text} | {m:.4f} ± {s:.4f} | {effect:+.4f} | {se2:.4f} | {"**yes**" if abs(effect) > se2 else "no"} |')


if __name__ == '__main__':
    main()
