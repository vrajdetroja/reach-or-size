"""Synthetic ring-size control (PREREGISTRATION.md 5.7): accuracy by pair x depth x RWSE-K, and the
two mechanism predictions checked cell by cell.

Usage: python -m analysis.synthetic   (writes results/synthetic/summary.csv and results/synthetic.md)

For pair (c, c'), c the smaller ring. Cells are means of test accuracy at best validation over seeds.
The preregistration says "near chance" and "high" without numbers. Fixed here, after the first 6 of
the 450 runs (pair (4, 6), L = 1) had been seen and before the rest (PREREGISTRATION.md 7):
CHANCE = mean below 0.60 (500 test graphs, so 0.60 is 4.5 binomial SDs above 0.5); HIGH = 0.90 or more.
- H-reach, no PE: chance for L < c/2, high for L >= c/2.
- H-reach, RWSE: high whenever K >= c, at any L.
- H-local: no PE stays at chance even at L >= c/2 + 2.
"""
import csv
import glob
import json
import os
import re
import statistics

from src.extension.run_synthetic import DEPTHS, KS
from src.extension.synthetic import PAIRS

CHANCE, HIGH = 0.60, 0.90
SEEDS = 3
NAME = re.compile(r'ring-(\d+)-(\d+)_L(\d+)_K(\d+)_s(\d+)$')


def load(folder='results/synthetic'):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, 'ring-*', 'summary.json'))):
        s = json.load(open(path))
        a, b, L, k, seed = (int(x) for x in NAME.match(s['run']).groups())
        with open(os.path.join(os.path.dirname(path), 'epochs.csv')) as f:
            last = list(csv.DictReader(f))[-1]
        rows.append(dict(run=s['run'], pair=f'{a}-{b}', depth=L, k=k, seed=seed, best_epoch=s['best_epoch'],
                         best_val=s['best_val'], test_at_best_val=s['test_at_best_val'],
                         final_train_acc=float(last['train_metric']), params=s['params']))
    return rows


def check(cells, c, condition, want):
    """Return (verdict, exceptions) over the cells selected by condition(L, k)."""
    sel = {key: v for key, v in cells.items() if condition(*key)}
    if not sel:
        return 'not testable in this grid', []
    ok = (lambda v: v < CHANCE) if want == 'chance' else (lambda v: v >= HIGH)
    bad = sorted(key for key, v in sel.items() if not ok(statistics.mean(v)))
    if not bad:
        return f'holds ({len(sel)}/{len(sel)} cells)', []
    return f'fails in {len(bad)}/{len(sel)} cells', bad


def main():
    rows = load()
    with open('results/synthetic/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = ['# Synthetic ring-size control (PREREGISTRATION.md 5.7)', '',
           f'{len(rows)} of {len(PAIRS) * len(DEPTHS) * len(KS) * SEEDS} runs. Mean test accuracy at best validation '
           f'over {SEEDS} seeds; chance is 0.5. Bands: chance < {CHANCE:.2f}, high ≥ {HIGH:.2f}. '
           'Train acc is the last epoch\'s training accuracy (train mode), mean over seeds. '
           '† marks a cell whose seeds disagree by more than 0.2 (some learn the task, others stay at chance).', '']
    out += ['## Smallest depth with mean accuracy ≥ 0.90', '',
            '| Rings | c/2 | ' + ' | '.join('no PE' if k == 0 else f'RWSE-{k}' for k in KS) + ' |',
            '| --- ' * (len(KS) + 2) + '|']
    for a, b in PAIRS:
        accs = {(r['depth'], r['k']): [] for r in rows if r['pair'] == f'{a}-{b}'}
        for r in rows:
            if r['pair'] == f'{a}-{b}':
                accs[(r['depth'], r['k'])].append(r['test_at_best_val'])
        vals = []
        for k in KS:
            hit = [L for L in DEPTHS if (L, k) in accs and statistics.mean(accs[(L, k)]) >= HIGH]
            if hit:
                vals.append(str(min(hit)))
            else:
                top = accs.get((DEPTHS[-1], k), [])
                won = sum(v >= 0.95 for v in top)
                vals.append(f'> {DEPTHS[-1]}' + (f' ({won} of {len(top)} seeds at L = {DEPTHS[-1]})' if won else ''))
        out.append(f'| {a} vs {b} | {a / 2:g} | ' + ' | '.join(vals) + ' |')
    out.append('')
    for a, b in PAIRS:
        pair = f'{a}-{b}'
        cells = {}
        train = {}
        for r in rows:
            if r['pair'] == pair:
                cells.setdefault((r['depth'], r['k']), []).append(r['test_at_best_val'])
                train.setdefault((r['depth'], r['k']), []).append(r['final_train_acc'])
        out += [f'## Rings {a} vs {b} (c = {a}, c/2 = {a / 2:g})', '',
                '| L | ' + ' | '.join('no PE' if k == 0 else f'RWSE-{k}' for k in KS) + ' | Train acc, no PE |',
                '| --- ' * (len(KS) + 2) + '|']
        for L in DEPTHS:
            vals = []
            for k in KS:
                v = cells.get((L, k), [])
                mark = '†' if v and max(v) - min(v) > 0.2 else ''
                vals.append('—' if not v else f"{statistics.mean(v):.3f}{mark}" + ('' if len(v) == SEEDS else f' ({len(v)})'))
            t = train.get((L, 0), [])
            out.append(f'| {L} | ' + ' | '.join(vals) + f" | {statistics.mean(t):.3f} |" if t else f'| {L} | ' + ' | '.join(vals) + ' | — |')
        out.append('')
        checks = [
            ('H-reach, no PE, L < c/2: chance', lambda L, k: k == 0 and L < a / 2, 'chance'),
            ('H-reach, no PE, L ≥ c/2: high', lambda L, k: k == 0 and L >= a / 2, 'high'),
            ('H-reach, RWSE with K ≥ c, any L: high', lambda L, k: k >= a, 'high'),
            ('Not preregistered: K < c and L < c/2: chance', lambda L, k: 0 < k < a and L < a / 2, 'chance'),
            ('H-local, no PE, L ≥ c/2 + 2: chance', lambda L, k: k == 0 and L >= a / 2 + 2, 'chance'),
        ]
        out += ['| Prediction | Verdict | Cells against it (L, K: accuracy) |', '| --- | --- | --- |']
        for label, cond, want in checks:
            verdict, bad = check(cells, a, cond, want)
            against = ', '.join(f'L{L} K{k}: {statistics.mean(cells[(L, k)]):.3f}' for L, k in bad) or '—'
            out.append(f'| {label} | {verdict} | {against} |')
        out.append('')
    text = '\n'.join(out)
    with open('results/synthetic.md', 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
