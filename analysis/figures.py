"""Every figure in REPORT.md and README.md, from committed results (CSV / JSON in results/).

Usage: python -m analysis.figures   (needs requirements-figures.txt; writes figures/*.png and *.pdf)

1. fig1_reproduction  per-number claims, our mean minus the paper's, in units of the paper's std
2. fig2_discrepancy   the paper's text vs the released code, GCN+ Peptides-func
3. fig3_extension     RWSE's gain across arms A-C, with what each hypothesis predicted
4. fig4_synthetic     synthetic ring-size task: accuracy by depth and RWSE walk length
5. fig5_depth         gain vs depth: DRew's digitized LapPE curves and our arm B

Colors are the validated reference palette (dataviz skill): series slots 1-3 checked all-pairs
for color-vision deficiency; the heatmap uses its single-hue blue ramp. Every figure has a table
twin in REPORT.md or results/.
"""
import csv
import math
import os
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from analysis.aggregate import CLAIMS  # noqa: E402
from analysis.discrepancy import ROWS, scores  # noqa: E402
from analysis.extension import NEW, REUSED, diff, load_cell, stats  # noqa: E402

OUT = 'figures'
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
WASH = '#f0efec'
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
        '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelsize': 9, 'axes.labelcolor': INK2, 'axes.edgecolor': BASE, 'axes.linewidth': 0.8,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'axes.grid': False, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'grid.linestyle': '-', 'legend.frameon': False, 'legend.fontsize': 8.5,
    'text.color': INK, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
})


def clean(ax, grid='y'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, axis=grid)
        ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'{OUT}/{name}.png')


def point(ax, x, y, yerr=None, xerr=None, color=BLUE, **kw):
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', color=color, ms=6.5, mec=SURFACE, mew=1.5,
                elinewidth=1.4, capsize=0, zorder=3, **kw)


def runs_by_config():
    groups = {}
    for r in csv.DictReader(open('results/all_runs.csv')):
        groups.setdefault(r['config'], []).append(float(r['test_at_best_val']))
    return groups


LABELS = {'C0': 'Tönshoff GCN · Peptides-func', 'C0b': 'OGB GCN · molhiv', 'C1': 'GCN+ · Peptides-func',
          'C2': 'GCN+ · Peptides-struct (MAE)', 'C3a': 'GatedGCN+ · Peptides-func', 'C3b': 'GatedGCN+ · Peptides-struct (MAE)',
          'C4a': 'GCN+ · molhiv', 'C4b': 'GatedGCN+ · molhiv'}


def fig1_reproduction():
    paper, groups = yaml.safe_load(open('configs/paper/in_scope.yaml')), runs_by_config()
    order = ['C0', 'C0b', 'C1', 'C2', 'C3a', 'C3b', 'C4a', 'C4b']
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(-2, 2, color=WASH, zorder=0, lw=0)
    ax.axvline(0, color=BASE, lw=0.8, zorder=1)
    for i, claim in enumerate(order):
        stem, (sec, key), tkey = CLAIMS[claim]
        t, xs = paper[sec][key][tkey], groups[stem]
        z = (statistics.mean(xs) - t['mean']) / t['std']
        point(ax, z, i, xerr=statistics.stdev(xs) / t['std'])
    ax.set_yticks(range(len(order)), [f'{c}  {LABELS[c]}' for c in order])
    ax.set_ylim(len(order) - 0.5, -1.1)
    ax.set_xlim(-3.8, 3.8)
    ax.set_xlabel("our mean − paper's mean, in units of the paper's std (bars: our std; for MAE, right is worse)")
    ax.text(0, -0.75, 'preregistered tolerance, ±2σ', ha='center', va='center', color=INK2, fontsize=8)
    ax.set_title('Every per-number claim lands inside the tolerance', pad=10)
    clean(ax, grid='x')
    ax.spines['left'].set_visible(False)
    save(fig, 'fig1_reproduction')


def fig2_discrepancy():
    base = scores('results/runs', 'gcnplus_peptides-func')
    mb, vb = statistics.mean(base), statistics.variance(base)
    rows = []
    for row, text, stem in ROWS:
        xs = scores('results/discrepancy', stem)
        rows.append((row, text, statistics.mean(xs) - mb, 2 * math.sqrt(vb / len(base) + statistics.variance(xs) / len(xs))))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (row, text, eff, two_se) in enumerate(rows):
        ax.barh(i, eff, height=0.42, color=BLUE, zorder=2)
        ax.errorbar(eff, i, xerr=two_se, fmt='none', ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
        ax.text(eff - two_se - 0.002, i, f'{eff:+.3f}'.replace('-', '−'), ha='right', va='center', color=INK, fontsize=8.5)
    ax.axvline(0, color=BASE, lw=0.8)
    ax.set_yticks(range(len(rows)), [f'{r}: {t}' for r, t, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.095, 0.006)
    ax.set_xlabel('change in test AP vs. the code-aligned model (0.7170); bars ±2·SE')
    ax.set_title("Most of the text-vs-code gap is unstated regularization", pad=10)
    clean(ax, grid='x')
    ax.spines['left'].set_visible(False)
    save(fig, 'fig2_discrepancy')


def extension_gains():
    cells = {k: load_cell(*v) for k, v in REUSED.items()}
    cells.update({k: load_cell('results/extension', v) for k, v in NEW.items()})
    s = {k: stats(v) for k, v in cells.items()}
    g = lambda on, off: diff(s[on], s[off])  # noqa: E731
    return dict(d0=g('K32', 'none'), A50=g('A50', 'A50_nope'), A25=g('A25', 'A25_nope'), B3=g('B3', 'B3_nope'),
                B6=g('B6', 'B6_nope'), K6=g('K6', 'none'), K16=g('K16', 'none'))


def fig3_extension():
    G = extension_gains()
    d0 = G['d0'][0]
    panels = [  # (title, x labels, gains, verdict, (x, y) of the preregistered prediction)
        ('A · training-set size', ['25%', '50%', '100%'], [G['A25'], G['A50'], G['d0']], 'absent', (0, 1.5 * d0)),
        ('B · depth, residuals on', ['3 layers', '6 layers'], [G['B3'], G['B6']], 'absent', (1, 0.5 * G['B3'][0])),
        ('C · RWSE walk length', ['K = 6', 'K = 16', 'K = 32'], [G['K6'], G['K16'], G['d0']], 'inconclusive', (0, 0.5 * d0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.3), sharey=True, gridspec_kw=dict(width_ratios=[3, 2.2, 3]))
    for ax, (title, xl, vals, verdict, (px, py)) in zip(axes, panels):
        ax.axhline(0, color=BASE, lw=0.8, zorder=1)
        ax.axhline(d0, color=MUTED, lw=0.8, zorder=1)
        xs = range(len(xl))
        ax.plot(xs, [v[0] for v in vals], color=BLUE, lw=1.5, zorder=2)
        point(ax, xs, [v[0] for v in vals], yerr=[1.96 * v[1] for v in vals])
        ax.hlines(py, px - 0.22, px + 0.22, color=ORANGE, lw=2.2, zorder=4)
        ax.set_xticks(list(xs), xl)
        ax.set_xlim(-0.5, len(xl) - 0.5)
        ax.set_title(title, pad=18)
        ax.text(0, 1.02, f'verdict: {verdict}', transform=ax.transAxes, color=INK2, fontsize=8, va='bottom')
        clean(ax)
    axes[0].set_ylabel('RWSE gain in test AP (RWSE − no PE), 95% CI')
    axes[0].text(1.3, d0 + 0.0006, 'Δ̂₀ = full-data gain', color=INK2, fontsize=7.5, va='bottom', ha='center')
    axes[0].set_ylim(-0.022, 0.032)
    handles = [plt.Line2D([], [], color=BLUE, marker='o', ms=6, mec=SURFACE, lw=1.5, label='measured RWSE gain, 95% CI'),
               plt.Line2D([], [], color=ORANGE, lw=2.2, label='preregistered prediction at that point: H-size, at least this '
                          '(A); H-reach, at most this (B: 0.5·g(3 layers); C: 0.5·Δ̂₀)')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.07, -0.13), ncol=1, fontsize=8)
    fig.suptitle("RWSE's gain on Peptides-func does not grow with less data or shrink with depth",
                 x=0.07, ha='left', fontsize=11, fontweight='bold', y=1.06)
    fig.subplots_adjust(wspace=0.12)
    save(fig, 'fig3_extension')


def fig4_synthetic():
    rows = list(csv.DictReader(open('results/synthetic/summary.csv')))
    pairs = [(4, 6), (5, 6), (8, 10), (12, 14), (16, 18)]
    depths, ks = [1, 2, 3, 4, 6, 8], [0, 4, 8, 16, 32]
    cmap = LinearSegmentedColormap.from_list('blue', RAMP)
    fig, axes = plt.subplots(1, len(pairs), figsize=(9.2, 2.9), sharey=True)
    for ax, (a, b) in zip(axes, pairs):
        grid = [[statistics.mean(float(r['test_at_best_val']) for r in rows
                                 if r['pair'] == f'{a}-{b}' and int(r['depth']) == L and int(r['k']) == k) for k in ks] for L in depths]
        mesh = ax.pcolormesh(grid, cmap=cmap, vmin=0.5, vmax=1.0, edgecolors=SURFACE, linewidth=1.5)
        boundary = sum(L < a / 2 for L in depths)  # rows below c/2
        ax.hlines(boundary, 0, len(ks), color=INK, lw=1.2)
        ax.text(len(ks) + 0.12, boundary, 'c/2', va='center', ha='left', color=INK2, fontsize=7.5)
        ax.set_xticks([i + 0.5 for i in range(len(ks))], ['none' if k == 0 else str(k) for k in ks])
        ax.set_yticks([i + 0.5 for i in range(len(depths))], [str(L) for L in depths])
        ax.set_title(f'rings {a} vs {b}', fontsize=9.5, pad=6)
        ax.set_xlabel('RWSE steps K')
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('layers L')
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label('test accuracy (chance = 0.5)', color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle('Without RWSE, GCN+ only learns small rings, and only once L ≥ c/2; long enough walks solve every pair at any depth',
                 x=0.07, ha='left', fontsize=10.5, fontweight='bold', y=1.04)
    save(fig, 'fig4_synthetic')


def fig5_depth():
    drew = {}
    for r in csv.DictReader(open('results/drew_digitized.csv')):
        drew[(r['pe'], r['series'], int(r['layers']))] = float(r['mean'])
    G = extension_gains()
    Ls = list(range(5, 24, 2))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.axhline(0, color=BASE, lw=0.8)
    for name, color in (('GCN', ORANGE), ('ResGCN', AQUA)):
        ys = [drew[('LapPE', name, L)] - drew[('none', name, L)] for L in Ls]
        ax.plot(Ls, ys, color=color, lw=1.5, marker='o', ms=5, mec=SURFACE, mew=1.2, label=f'DRew {name}, LapPE gain')
        ax.text(Ls[-1] + 0.4, ys[-1], f'DRew {name} + LapPE', va='center', color=INK2, fontsize=8)
    ours = [G['B3'], G['B6']]
    ax.plot([3, 6], [v[0] for v in ours], color=BLUE, lw=1.5, zorder=3)
    point(ax, [3, 6], [v[0] for v in ours], yerr=[1.96 * v[1] for v in ours], label='ours: GCN+ residual, RWSE-32 gain (95% CI)')
    ax.text(6.4, ours[1][0], 'ours, GCN+ + RWSE', va='center', color=INK2, fontsize=8)
    ax.set_xlim(2, 28.5)
    ax.set_xticks([3, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23])
    ax.set_xlabel('layers')
    ax.set_ylabel('PE gain in test AP (PE − no PE)')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.set_title("Peptides-func: DRew's LapPE gain shrinks from 5 to 7 layers; ours grows from 3 to 6", pad=10)
    clean(ax)
    save(fig, 'fig5_depth')


def main():
    fig1_reproduction()
    fig2_discrepancy()
    fig3_extension()
    fig4_synthetic()
    fig5_depth()


if __name__ == '__main__':
    main()
