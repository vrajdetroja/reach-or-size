"""Extension thresholds and power with our measured numbers (PREREGISTRATION.md 5.2, 5.5, 5.6).

The rules themselves are fixed (5.5, 6). This restates them with the measured Delta_0 and seed stds
in place of the paper's, so the arms' realistic outcomes are on record before any extension run.

Inputs: results/all_runs.csv, C1 (gcnplus_peptides-func) and C5a off (gcnplus_peptides-func_nope).
Assumption, stated in the output: every new cell has the std of its PE-matched reproduction cell
(RWSE on -> C1's, RWSE off -> C5a off's). Arm A's subsets and arm B's residual configs may be noisier.

Usage: python -m analysis.thresholds   (prints markdown; REPORT.md 7.1 quotes it)
"""
import csv
import math
import statistics


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def cell(rows, config):
    v = [float(r['test_at_best_val']) for r in rows if r['config'] == config]
    assert len(v) == 3, (config, len(v))
    return statistics.mean(v), statistics.stdev(v)


def main():
    rows = list(csv.DictReader(open('results/all_runs.csv')))
    mu_on, s_on = cell(rows, 'gcnplus_peptides-func')
    mu_off, s_off = cell(rows, 'gcnplus_peptides-func_nope')
    n = 3
    d0 = mu_on - mu_off
    m = 0.5 * d0
    se_gain = math.sqrt(s_on ** 2 / n + s_off ** 2 / n)
    print('## Measured inputs\n')
    print(f'- RWSE on (C1): {mu_on:.4f} ± {s_on:.4f}; RWSE off (C5a): {mu_off:.4f} ± {s_off:.4f} (3 seeds, sample std)')
    print(f'- Δ̂₀ = {d0:.4f} (paper 0.0192), so m = 0.5·Δ̂₀ = {m:.4f}; SE(Δ̂₀) = {se_gain:.4f}, Δ̂₀/SE = {d0 / se_gain:.1f}')
    print('- Precondition (§5.2): C5a reproduced, so the extension may start.\n')
    # Arms A and B: difference of two gains, four cells. Arm C: difference of two RWSE-on means.
    sp = 0.0081  # the paper's pooled per-run std (5.6), as a less optimistic case
    cases = [('A, B: measured stds', math.sqrt(2 * s_on ** 2 / n + 2 * s_off ** 2 / n)),
             ('A, B: paper std 0.0081', math.sqrt(4 * sp ** 2 / n)),
             ('C: K = 6 as stable as K = 32', math.sqrt(2 * s_on ** 2 / n)),
             ('C: K = 6 as noisy as no PE', math.sqrt(s_on ** 2 / n + s_off ** 2 / n)),
             ('C: paper std 0.0081', math.sqrt(2 * sp ** 2 / n))]
    print('## What each rule needs at 3 seeds\n')
    print('| Arm, std assumption | SE of X | Positive needs X̂ > | Power at X = m | Power at X = Δ̂₀ '
          '| P("absent") if X = 0 | Seeds/cell for 80% power at X = m |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for name, se in cases:
        need = math.ceil(n * (se * (2 + 0.8416) / m) ** 2)  # SE ~ 1/sqrt(seeds); 0.8416 = z for 80% power
        print(f'| {name} | {se:.4f} | {2 * se:.4f} | {phi(m / se - 2):.2f} | {phi(d0 / se - 2):.2f} '
              f'| {phi(m / se - 1.645):.2f} | {need} |')
    se_ab = cases[0][1]
    se_g6 = math.sqrt(s_on ** 2 / n + s_off ** 2 / n)  # K = 6 assumed as stable as C1
    print(f'\nH-local pattern (§5.5): needs ĝ(6) − 2·SE > 0, i.e. ĝ(6) > {2 * se_g6:.4f} ({2 * se_g6 / d0:.0%} of Δ̂₀), '
          'and arm C absent.')
    print('\nPower is for the positive rule, P(X̂ − 2·SE > 0). "Absent" is X̂ + 1.645·SE < m. Arm B\'s m is 0.5·ĝ(L=3) '
          'with residuals on, unknown until its runs exist; the table uses m = 0.5·Δ̂₀ for it.')
    print(f'\nExtra-seed rule check (§5.6): with 5 seeds the arm A/B SE becomes {se_ab * math.sqrt(3 / 5):.4f}.')


if __name__ == '__main__':
    main()
