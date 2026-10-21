"""Digitize DRew (Gutteridge et al., ICML 2023) App. B.5, Figs. 4-5: Peptides-func test AP vs depth
for GCN and residual GCN with 500k parameters, without and with LapPE (PREREGISTRATION.md 5.8).

Usage: python -m analysis.drew_digitize SVG_DIR   (SVG_DIR holds ablation_no_pe.svg and ablation_lappe.svg
       from https://arxiv.org/html/2305.08018v2/)

The SVGs (PDF-converted Plotly figures) draw each point's error bar as a vertical segment centred on
the mean, so the segment midpoint is the plotted mean and its half-length the plotted std. Axes are
taken from the gridlines: 8 horizontal lines at AP 0.58..0.72 and vertical lines at depth 5, 10,
15, 20 (read off the rendered figures). Series follow legend order (blue solid / dashed / dotted for the
three DRew variants, red solid GCN, red dashed ResGCN); the check below confirms it by matching the two
solid polylines to the first and fourth groups of bars. Writes results/drew_digitized.csv.
"""
import csv
import re
import sys

SERIES = ['nu1_DRew-GCN', 'nuL2_DRew-GCN', 'DRew-GCN', 'GCN', 'ResGCN']  # legend order = drawing order
Y_TICKS = [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72]  # bottom to top
X_TICKS = [5, 10, 15, 20]


def attrs(tag):
    return dict(re.findall(r'([a-zA-Z:-]+)="([^"]*)"', tag))


def affine(t):
    a, b, c, d, e, f = (float(x) for x in re.findall(r'-?[\d.]+(?:e-?\d+)?', t))
    return lambda x, y: (a * x + c * y + e, b * x + d * y + f)


def digitize(path):
    s = open(path).read()
    tags = [attrs(m.group(1)) for m in re.finditer(r'<path([^>]*)/?>', s)]
    hgrid, vgrid, bars, solid = [], [], [], {}
    for a in tags:
        d, t = a.get('d', ''), a.get('transform', '')
        if a.get('stroke') == '#808080' and re.fullmatch(r'M80 0H\d+', d):
            hgrid.append(affine(t)(0, 0)[1])
        elif a.get('stroke') == '#808080' and re.fullmatch(r'M0 60V520', d):
            vgrid.append(affine(t)(0, 0)[0])
        elif a.get('stroke') in ('#636efa', '#ef553b') and re.fullmatch(r'M[\d. ]+', d):
            pts = [float(v) for v in d[1:].split()]
            solid[a['stroke']] = [affine(t)(x, y) for x, y in zip(pts[::2], pts[1::2])]
        else:
            m = re.fullmatch(r'M[\d.]+ ([\d.]+)H[\d.]+M([\d.]+) [\d.]+V([\d.]+)M[\d.]+ [\d.]+H[\d.]+', d)
            if m and 'fill' not in a:
                f = affine(t)
                x, y1 = f(float(m.group(2)), float(m.group(1)))
                _, y2 = f(float(m.group(2)), float(m.group(3)))
                bars.append((x, (y1 + y2) / 2, abs(y2 - y1) / 2))
    hgrid.sort(reverse=True)   # bottom (largest page y) first -> 0.58
    vgrid.sort()
    assert len(hgrid) == len(Y_TICKS) and len(vgrid) == len(X_TICKS) and len(bars) == 50, (len(hgrid), len(vgrid), len(bars))
    for color, k in (('#636efa', 0), ('#ef553b', 3)):  # nu1 DRew-GCN and GCN are the solid lines
        assert len(solid[color]) == 10 and all(abs(x - bx) < 0.01 and abs(y - by) < 0.01
                   for (x, y), (bx, by, _) in zip(solid[color], bars[10 * k:10 * k + 10])), (path, color)
    y_per = (hgrid[0] - hgrid[-1]) / (Y_TICKS[-1] - Y_TICKS[0])
    x_per = (vgrid[-1] - vgrid[0]) / (X_TICKS[-1] - X_TICKS[0])
    rows = []
    for k, name in enumerate(SERIES):
        for x, ymid, half in bars[10 * k:10 * k + 10]:
            depth = X_TICKS[0] + (x - vgrid[0]) / x_per
            rows.append(dict(series=name, layers=round(depth), mean=Y_TICKS[0] + (hgrid[0] - ymid) / y_per,
                             std=half / y_per))
    return sorted(rows, key=lambda r: (SERIES.index(r['series']), r['layers']))


def main():
    folder = sys.argv[1]
    out = []
    for pe, name in (('none', 'ablation_no_pe.svg'), ('LapPE', 'ablation_lappe.svg')):
        for r in digitize(f'{folder}/{name}'):
            out.append(dict(pe=pe, **r))
    with open('results/drew_digitized.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pe', 'series', 'layers', 'mean', 'std'])
        w.writeheader()
        for r in out:
            w.writerow({**r, 'mean': f"{r['mean']:.4f}", 'std': f"{r['std']:.4f}"})
    get = {(r['pe'], r['series'], r['layers']): r for r in out}
    print('# DRew App. B.5 digitized: Peptides-func test AP, 500k params (results/drew_digitized.csv)\n')
    print('| Layers | GCN | GCN+LapPE | LapPE gain | ResGCN | ResGCN+LapPE | LapPE gain |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for L in range(5, 24, 2):
        g0, g1 = get[('none', 'GCN', L)], get[('LapPE', 'GCN', L)]
        r0, r1 = get[('none', 'ResGCN', L)], get[('LapPE', 'ResGCN', L)]
        print(f"| {L} | {g0['mean']:.3f} ± {g0['std']:.3f} | {g1['mean']:.3f} ± {g1['std']:.3f} | {g1['mean'] - g0['mean']:+.3f} "
              f"| {r0['mean']:.3f} ± {r0['std']:.3f} | {r1['mean']:.3f} ± {r1['std']:.3f} | {r1['mean'] - r0['mean']:+.3f} |")
    print()
    for name in ('GCN', 'ResGCN'):
        gain = {L: get[('LapPE', name, L)]['mean'] - get[('none', name, L)]['mean'] for L in range(5, 24, 2)}
        deep = [gain[L] for L in range(7, 24, 2)]
        print(f'- {name}: LapPE gain {gain[5]:+.3f} at L = 5, {gain[7]:+.3f} at L = 7 ({gain[7] / gain[5] - 1:+.0%}); '
              f'mean {sum(deep) / len(deep):+.3f} over L = 7..23 (range {min(deep):+.3f} to {max(deep):+.3f})')


if __name__ == '__main__':
    main()
