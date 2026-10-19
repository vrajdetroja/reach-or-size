"""Environment checks (PREREGISTRATION.md 3.6): the baseline and the authors' code in our environment.

Usage: python -m analysis.env_checks [--runs results/env_checks/runs] [--code-runs results/env_checks/runs_code] [--theirs results/env_checks/theirs]

Check 1 (baseline): Tönshoff-config GCN on Peptides-func, 3 seeds, our implementation; the mean
test AP must lie in [0.6760, 0.6960].
Check 2 (harness): GCN+ on Peptides-func, 1 seed with the authors' code; test AP must lie in
[0.7127, 0.7395]. Our paper-literal and code-aligned GCN+ seeds are reported next to it.
Test AP is always taken at the best-validation epoch.
"""
import argparse
import glob
import json
import os
import statistics

C0_RANGE = (0.6760, 0.6960)
C1_RANGE = (0.7127, 0.7395)


def ours(run_dir):
    path = os.path.join(run_dir, 'summary.json')
    if not os.path.exists(path):
        return None
    s = json.load(open(path))
    return s['test_at_best_val'], s['epochs_run']


def theirs(out_dir):
    """Test AP at the best-validation epoch from GraphGym's val/test stats.json."""
    found = {}
    for split in ('val', 'test'):
        files = glob.glob(os.path.join(out_dir, '**', split, 'stats.json'), recursive=True)
        if not files:
            return None
        found[split] = [json.loads(line) for line in open(files[0]) if line.strip()]
    best = max(range(len(found['val'])), key=lambda i: found['val'][i]['ap'])
    return found['test'][best]['ap'], len(found['val'])


def inside(x, rng):
    return rng[0] <= x <= rng[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='results/env_checks/runs')
    parser.add_argument('--code-runs', default='results/env_checks/runs_code')
    parser.add_argument('--theirs', default='results/env_checks/theirs')
    args = parser.parse_args()

    print('# Environment checks (PREREGISTRATION.md 3.6)\n')
    c0 = [ours(os.path.join(args.runs, f'tonshoff-gcn_peptides-func_s{s}')) for s in (0, 1, 2)]
    c0 = [r for r in c0 if r is not None]
    check1 = None
    if len(c0) == 3:
        aps = [ap for ap, _ in c0]
        mean = statistics.mean(aps)
        check1 = inside(mean, C0_RANGE)
        print(f'**Check 1, baseline (C0).** Tönshoff GCN, 3 seeds: {", ".join(f"{a:.4f}" for a in aps)}. '
              f'Mean {mean:.4f} ± {statistics.stdev(aps):.4f}; target 0.6860, pass range {C0_RANGE}. '
              f'**{"PASS" if check1 else "FAIL"}**.\n')
    else:
        print(f'**Check 1, baseline (C0).** Incomplete: {len(c0)} of 3 seeds found.\n')

    t = theirs(os.path.join(args.theirs, 'gcn_peptides-func_s0'))
    check2 = None
    rows = []
    if t is not None:
        check2 = inside(t[0], C1_RANGE)
        rows.append(('Authors\' code (GNNPlus 0e02ad9)', t))
    for label, d in (('Ours, code-aligned (configs/code)', os.path.join(args.code_runs, 'gcnplus_peptides-func_s0')),
                     ('Ours, paper-literal (configs/runs)', os.path.join(args.runs, 'gcnplus_peptides-func_s0'))):
        r = ours(d)
        if r is not None:
            rows.append((label, r))
    print('**Check 2, harness.** GCN+ on Peptides-func, seed 0. Target 0.7261, pass range '
          f'{C1_RANGE} (applies to the authors\' code).\n')
    print('| Implementation | Test AP at best val | Epochs |')
    print('| --- | --- | --- |')
    for label, (ap, epochs) in rows:
        print(f'| {label} | {ap:.4f} | {epochs} |')
    if check2 is None:
        print('\nAuthors\' run not found.')
    else:
        print(f'\nAuthors\' code: **{"PASS" if check2 else "FAIL"}**.')

    print('\n**Outcome.**', end=' ')
    if check1 is None or check2 is None:
        print('Incomplete: some runs are missing.')
    elif check1 and check2:
        print('Both checks pass.')
    elif check2 and not check1:
        print('The baseline fails while the authors\' code reproduces: the baseline implementation is at fault.')
    else:
        print('The authors\' code misses its own number: our data pipeline or package versions differ from theirs. '
              'Fix that before anything else.')


if __name__ == '__main__':
    main()
