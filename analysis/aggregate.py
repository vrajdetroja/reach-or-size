"""Collect run summaries into results/all_runs.csv and check claims against tolerance.

Usage: python -m analysis.aggregate [--runs results/runs]

Tolerance rule (PREREGISTRATION.md 3.1): reproduced if |our mean - paper mean| <= 2 x paper std.
PE rule (3.4): our drop > 2 * sqrt(s_on^2/n_on + s_off^2/n_off).
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

import yaml

from src.data.datasets import task_info

# Claim -> (run config stem, paper section in configs/paper/in_scope.yaml, target key)
CLAIMS = {
    'C1': ('gcnplus_peptides-func', ('gcn_plus', 'peptides-func'), 'target'),
    'C2': ('gcnplus_peptides-struct', ('gcn_plus', 'peptides-struct'), 'target'),
    'C3a': ('gatedgcnplus_peptides-func', ('gatedgcn_plus', 'peptides-func'), 'target'),
    'C3b': ('gatedgcnplus_peptides-struct', ('gatedgcn_plus', 'peptides-struct'), 'target'),
    'C4a': ('gcnplus_ogbg-molhiv', ('gcn_plus', 'ogbg-molhiv'), 'target'),
    'C4b': ('gatedgcnplus_ogbg-molhiv', ('gatedgcn_plus', 'ogbg-molhiv'), 'target'),
    'C0': ('tonshoff-gcn_peptides-func', ('baselines', 'gcn_peptides-func'), 'target'),
    'C0b': ('ogb-gcn_ogbg-molhiv', ('baselines', 'gcn_ogbg-molhiv'), 'target'),
}
PE_CLAIMS = {'C5a': ('gcnplus_peptides-func', 'gcnplus_peptides-func_nope'),
             'C5b': ('gcnplus_ogbg-molhiv', 'gcnplus_ogbg-molhiv_nope')}


def load_runs(runs_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, '*', 'summary.json'))):
        with open(path) as f:
            s = json.load(f)
        if s.get('smoke_test'):
            continue
        s['config'] = re.sub(r'_s\d+(_.*)?$', '', s['run'])
        rows.append(s)
    return rows


def by_config(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r['config'], []).append(r['test_at_best_val'])
    return groups


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float('nan'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='results/runs')
    parser.add_argument('--csv', default='results/all_runs.csv')
    args = parser.parse_args()
    rows = load_runs(args.runs)
    fields = ['run', 'config', 'dataset', 'arch', 'seed', 'params', 'epochs_run', 'best_epoch', 'best_val',
              'test_at_best_val', 'test_at_last_epoch', 'mean_seconds_per_epoch', 'device', 'torch']
    with open(args.csv, 'w') as f:
        f.write(','.join(fields) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(k, '')) for k in fields) + '\n')
    paper = yaml.safe_load(open('configs/paper/in_scope.yaml'))
    groups = by_config(rows)
    print(f'{len(rows)} runs -> {args.csv}\n')
    print('| Claim | Paper | Ours (mean ± std) | n | |Δ| | Tolerance | Verdict |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for claim, (stem, (sec, key), tkey) in CLAIMS.items():
        target = paper[sec][key][tkey]
        xs = groups.get(stem, [])
        if not xs:
            print(f'| {claim} | {target["mean"]:.4f} ± {target["std"]:.4f} | — | 0 | | | not run |')
            continue
        m, s = mean_std(xs)
        delta, tol = abs(m - target['mean']), 2 * target['std']
        verdict = 'within tolerance' if delta <= tol else 'outside tolerance'
        print(f'| {claim} | {target["mean"]:.4f} ± {target["std"]:.4f} | {m:.4f} ± {s:.4f} | {len(xs)} | '
              f'{delta:.4f} | {tol:.4f} | {verdict} |')
    print('\n| Claim | Drop (ours) | 2·SE | Verdict |')
    print('| --- | --- | --- | --- |')
    for claim, (on, off) in PE_CLAIMS.items():
        a, b = groups.get(on, []), groups.get(off, [])
        if len(a) < 2 or len(b) < 2:
            print(f'| {claim} | — | — | needs ≥ 2 seeds per cell |')
            continue
        (ma, sa), (mb, sb) = mean_std(a), mean_std(b)
        dataset = next(r['dataset'] for r in rows if r['config'] == on)
        drop = (ma - mb) if task_info(dataset)['higher_is_better'] else (mb - ma)
        se = math.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
        print(f'| {claim} | {drop:.4f} | {2 * se:.4f} | {"reproduced" if drop > 2 * se else "not reproduced"} |')


if __name__ == '__main__':
    main()
