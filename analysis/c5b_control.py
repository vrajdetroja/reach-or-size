"""C5b failure check (PREREGISTRATION.md 3.3): the authors' code on the same claim.

Usage: python -m analysis.c5b_control [--theirs results/theirs_c5b] [--runs results/runs]

C5b: removing RWSE hurts GCN+ on ogbg-molhiv (paper 0.8012 -> 0.7593, drop 0.0419). The PE rule
(3.4) needs drop > 2 * SE. We apply the same rule to the authors' code (GNNPlus 0e02ad9, 3 seeds
per variant) and set it next to ours. Test AUROC is taken at the best-validation epoch.
"""
import argparse
import glob
import json
import math
import os
import statistics


def best_val_test(out_dir, metric):
    found = {}
    for split in ('val', 'test'):
        files = glob.glob(os.path.join(out_dir, '**', split, 'stats.json'), recursive=True)
        if not files:
            return None
        found[split] = [json.loads(line) for line in open(files[0]) if line.strip()]
    best = max(range(len(found['val'])), key=lambda i: found['val'][i][metric])
    return found['test'][best][metric]


def summarize(xs):
    return statistics.mean(xs), statistics.stdev(xs), len(xs)


def drop_row(label, on, off):
    (ma, sa, na), (mb, sb, nb) = summarize(on), summarize(off)
    drop = ma - mb
    se2 = 2 * math.sqrt(sa ** 2 / na + sb ** 2 / nb)
    ok = drop > se2
    print(f'| {label} | {ma:.4f} ± {sa:.4f} | {mb:.4f} ± {sb:.4f} | {drop:+.4f} | {se2:.4f} | {"yes" if ok else "no"} |')
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theirs', default='results/theirs_c5b')
    parser.add_argument('--runs', default='results/runs')
    args = parser.parse_args()
    theirs = {v: [x for x in (best_val_test(os.path.join(args.theirs, f'gcn_hiv_{v}_s{s}'), 'auc') for s in (0, 1, 2))
                  if x is not None] for v in ('on', 'off')}
    ours = {v: [json.load(open(f))['test_at_best_val'] for f in
                sorted(glob.glob(os.path.join(args.runs, f'gcnplus_ogbg-molhiv{"" if v == "on" else "_nope"}_s*', 'summary.json')))]
            for v in ('on', 'off')}
    print('# C5b control: RWSE ablation of GCN+ on ogbg-molhiv (test AUROC at best val)\n')
    print('| Implementation | With RWSE | Without RWSE | Drop | 2·SE | Drop > 2·SE? |')
    print('| --- | --- | --- | --- | --- | --- |')
    print('| Paper (Tables 4, 6; 5 seeds) | 0.8012 ± 0.0124 | 0.7593 ± 0.0051 | +0.0419 | — | — |')
    ours_ok = drop_row('Ours, code-aligned (RTX 4090)', ours['on'], ours['off']) if min(map(len, ours.values())) >= 2 else None
    if min(map(len, theirs.values())) < 2:
        print(f'\nAuthors\' runs incomplete: {[len(v) for v in theirs.values()]} seeds.')
        return
    theirs_ok = drop_row('Authors\' code (GNNPlus 0e02ad9)', theirs['on'], theirs['off'])
    print('\n**Verdict (PREREGISTRATION.md 3.3, 3.4).**', end=' ')
    if theirs_ok and not ours_ok:
        print('The authors\' code reproduces the drop and ours does not: look for the cause in our implementation.')
    elif theirs_ok and ours_ok:
        print('Both implementations reproduce the drop.')
    elif not theirs_ok and not ours_ok:
        print('Neither implementation shows a drop beyond 2·SE at 3 seeds. With the failure checks done, '
              'C5b is not reproduced at 3 seeds; report the effect sizes and the seeds a decisive test would need.')
    else:
        print('Ours reproduces the drop but the authors\' code does not at 3 seeds; report both.')


if __name__ == '__main__':
    main()
