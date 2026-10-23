"""Extension arms A-C: contrasts, decision rules and joint reading (PREREGISTRATION.md 5.2-5.6).

Written and committed before any extension run. Test AP at best validation, every seed present per
cell (3, or 5 for an arm the extra-seed rule extended), sample std, cells treated as independent.

- Delta_0 = mean(C1) - mean(C5a): the reproduction's RWSE-32 and no-RWSE cells (results/runs).
- Arm A: X_A = g(25%) - g(100%), m = 0.5 Delta_0. g(100%) is Delta_0 itself.
- Arm B: X_B = g(L=3) - g(L=6), residuals on in both, m = 0.5 g(L=3).
- Arm C: X_C = mean(K=32) - mean(K=6), m = 0.5 Delta_0; K = 32 is C1. Also g(6) = mean(K=6) - mean(C5a).
- Positive: X - 2 SE > 0. Absent: X + 1.645 SE < m. Otherwise inconclusive.
- Extra-seed rule (5.6): an arm with SE < X < 2 SE gets 2 more seeds, priority B, then A.

Usage: python -m analysis.extension   (writes results/extension.md)
"""
import glob
import json
import math
import os
import statistics

REUSED = {'K32': ('results/runs', 'gcnplus_peptides-func'), 'none': ('results/runs', 'gcnplus_peptides-func_nope')}
NEW = {
    'A25': 'gcnplus_peptides-func_armA_frac25', 'A25_nope': 'gcnplus_peptides-func_armA_frac25_nope',
    'A50': 'gcnplus_peptides-func_armA_frac50', 'A50_nope': 'gcnplus_peptides-func_armA_frac50_nope',
    'B3': 'gcnplus_peptides-func_armB_L3res', 'B3_nope': 'gcnplus_peptides-func_armB_L3res_nope',
    'B6': 'gcnplus_peptides-func_armB_L6res', 'B6_nope': 'gcnplus_peptides-func_armB_L6res_nope',
    'K6': 'gcnplus_peptides-func_armC_rwse6', 'K16': 'gcnplus_peptides-func_armC_rwse16',
}


def load_cell(folder, config):
    """Test AP at best validation for every seed of one config (runs named <config>_s<seed>)."""
    vals = []
    for path in sorted(glob.glob(os.path.join(folder, f'{config}_s*', 'summary.json'))):
        run = os.path.basename(os.path.dirname(path))
        if run[len(config) + 2:].isdigit():  # skip tagged runs such as _s0_smoke
            vals.append(json.load(open(path))['test_at_best_val'])
    return vals


def stats(v):
    return dict(mean=statistics.mean(v), var=statistics.variance(v), n=len(v))


def diff(a, b):
    """Difference of two cell means and its SE."""
    return a['mean'] - b['mean'], math.sqrt(a['var'] / a['n'] + b['var'] / b['n'])


def diff_of_gains(on1, off1, on2, off2):
    """(on1 - off1) - (on2 - off2) and its SE, four independent cells."""
    x = (on1['mean'] - off1['mean']) - (on2['mean'] - off2['mean'])
    return x, math.sqrt(sum(c['var'] / c['n'] for c in (on1, off1, on2, off2)))


def verdict(x, se, m):
    if x - 2 * se > 0:
        return 'positive'
    if x + 1.645 * se < m:
        return 'absent'
    return 'inconclusive'


def joint_reading(a, b, c, g6_positive):
    """Every row of the 5.5 joint-reading table whose pattern holds; 'Inconclusive' if none does."""
    rows = []
    if a == 'positive' and b != 'positive' and c != 'positive':
        rows.append('Evidence for H-size')
    if (b == 'positive' or c == 'positive') and a == 'absent':
        rows.append('Evidence for H-reach; the size explanation does not hold on Peptides-func')
    if c == 'absent' and g6_positive and b != 'positive':
        rows.append('Evidence for H-local: the gain comes from local structure message passing does not compute')
    if a == 'positive' and (b == 'positive' or c == 'positive'):
        rows.append('RWSE does more than one job; we quantify each')
    return rows or ['Inconclusive with the seeds run; we report effect sizes, CIs and the seeds needed']


def extra_seed_candidates(arms):
    """Arms with SE < X < 2 SE, in the 5.6 priority order (B, then A; C is not extended)."""
    return [name for name in ('B', 'A') if name in arms and arms[name]['se'] < arms[name]['x'] < 2 * arms[name]['se']]


def main():
    cells = {k: load_cell(*v) for k, v in REUSED.items()}
    cells.update({k: load_cell('results/extension', v) for k, v in NEW.items()})
    out = ['# Extension arms A-C (PREREGISTRATION.md 5)', '',
           'Test AP at best validation. Positive: X − 2·SE > 0. Absent: X + 1.645·SE < m. Otherwise inconclusive.', '',
           '| Cell | Seeds | Test AP |', '| --- | --- | --- |']
    for k, v in cells.items():
        out.append(f'| {k} | {len(v)} | ' + (f'{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}' if len(v) > 1 else
                                              ('—' if not v else f'{v[0]:.4f}')) + ' |')
    missing = [k for k, v in cells.items() if len(v) < 2]
    if missing:
        out += ['', f'**Not enough seeds yet for: {", ".join(missing)}.** Contrasts need at least 2 seeds per cell.']
        print('\n'.join(out))
        return
    s = {k: stats(v) for k, v in cells.items()}
    d0, se_d0 = diff(s['K32'], s['none'])
    m = 0.5 * d0
    arms = {}
    x, se = diff_of_gains(s['A25'], s['A25_nope'], s['K32'], s['none'])
    arms['A'] = dict(x=x, se=se, m=m, label='g(25%) − g(100%)')
    g3, se_g3 = diff(s['B3'], s['B3_nope'])
    x, se = diff_of_gains(s['B3'], s['B3_nope'], s['B6'], s['B6_nope'])
    arms['B'] = dict(x=x, se=se, m=0.5 * g3, label='g(L=3) − g(L=6)')
    x, se = diff(s['K32'], s['K6'])
    arms['C'] = dict(x=x, se=se, m=m, label='mean(K=32) − mean(K=6)')
    for a in arms.values():
        a['verdict'] = verdict(a['x'], a['se'], a['m'])
    g6, se_g6 = diff(s['K6'], s['none'])
    g6_positive = g6 - 2 * se_g6 > 0
    gains = [('g(100%) = Δ̂₀', d0, se_d0), ('g(50%)', *diff(s['A50'], s['A50_nope'])), ('g(25%)', *diff(s['A25'], s['A25_nope'])),
             ('g(L=3), residual', g3, se_g3), ('g(L=6), residual', *diff(s['B6'], s['B6_nope'])),
             ('g(K=6)', g6, se_g6), ('g(K=16)', *diff(s['K16'], s['none'])), ('g(K=32) = Δ̂₀', d0, se_d0)]
    out += ['', '## PE gains', '', '| Gain | Estimate | SE | 95% CI |', '| --- | --- | --- | --- |']
    out += [f'| {n} | {g:+.4f} | {e:.4f} | [{g - 1.96 * e:+.4f}, {g + 1.96 * e:+.4f}] |' for n, g, e in gains]
    out += ['', '## Decision rules (§5.5)', '', '| Arm | X | Estimate | SE | m | X − 2·SE | X + 1.645·SE | Verdict |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, a in arms.items():
        out.append(f"| {name} | {a['label']} | {a['x']:+.4f} | {a['se']:.4f} | {a['m']:+.4f} | {a['x'] - 2 * a['se']:+.4f} "
                   f"| {a['x'] + 1.645 * a['se']:+.4f} | **{a['verdict']}** |")
    if arms['B']['m'] <= 0:
        out.append('\nArm B: ĝ(L=3) ≤ 0, so H-reach predicts no positive effect to lose; m_B is not meaningful.')
    seeds = min(len(v) for v in cells.values())
    open_arms = [(name, a) for name, a in arms.items() if a['verdict'] == 'inconclusive']
    if open_arms:
        # 5.5's "Anything else" row reports the seeds needed: seeds per cell at which the current estimate
        # would cross each bound, if it and the per-cell stds stayed as measured (SE ~ 1/sqrt(seeds)).
        out += ['', '## Seeds needed for the inconclusive arms', '',
                f'Seeds per cell at which the current estimate would cross each bound, if it and the per-cell stds '
                f'stayed as measured ({seeds} seeds now).', '', '| Arm | To call it positive | To call it absent |', '| --- | --- | --- |']
        for name, a in open_arms:
            pos = math.ceil(seeds * (2 * a['se'] / a['x']) ** 2) if a['x'] > 0 else 'never at this estimate'
            ab = math.ceil(seeds * (1.645 * a['se'] / (a['m'] - a['x'])) ** 2) if a['m'] > a['x'] else 'never at this estimate'
            out.append(f'| {name} | {pos} | {ab} |')
    out += ['', f"H-local's extra condition: ĝ(6) − 2·SE = {g6 - 2 * se_g6:+.4f} ({'met' if g6_positive else 'not met'}).", '',
            '## Joint reading (§5.5)', '']
    out += [f'- {r}' for r in joint_reading(arms['A']['verdict'], arms['B']['verdict'], arms['C']['verdict'], g6_positive)]
    seeds_now = min(len(v) for k, v in cells.items() if k in NEW)
    cands = extra_seed_candidates(arms) if seeds_now < 5 else []
    out += ['', '## Extra-seed rule (§5.6)', '',
            f'Candidates (SE < X < 2·SE, priority B then A): {", ".join(cands) if cands else "none"}.']
    text = '\n'.join(out)
    with open('results/extension.md', 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
