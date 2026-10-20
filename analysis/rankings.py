"""Ranking claims R1-R3 under the preregistered distinguishability rule (CLAIMS.md).

Usage: python -m analysis.rankings [--runs results/runs]

A gap to a competitor counts as real only if |gap| > 2 * sqrt(s_ours^2 / n_ours + s_comp^2 / n_comp).
n_comp is the competitor's seed count; it is not transcribed here, so we use n_comp = 1, the
preregistered conservative default, which can only make gaps look smaller.
Competitor numbers are transcribed from GNN+ Tables 3-4 (arXiv 2502.09263v3), as the paper cites them.
"""
import argparse
import glob
import json
import math
import os
import statistics

COMPETITORS = {
    'peptides-func': dict(higher=True, rows={
        'GT': (0.6326, 0.0126), 'SAN': (0.6439, 0.0075), 'GraphGPS': (0.6535, 0.0041), 'GraphGPS (2023)': (0.6534, 0.0091),
        'MGT': (0.6817, 0.0064), 'DRew': (0.7150, 0.0044), 'Graph ViT/MLP-Mixer': (0.6970, 0.0080),
        'Exphormer': (0.6258, 0.0092), 'GRIT': (0.6988, 0.0082), 'Subgraphormer': (0.6415, 0.0052),
        'GRED': (0.7133, 0.0011), 'GEAET': (0.6485, 0.0035), 'TIGT': (0.6679, 0.0074), 'GECO': (0.6975, 0.0025),
        'GPNN': (0.6955, 0.0057), 'Graph-Mamba': (0.6739, 0.0087), 'GSSC': (0.7081, 0.0062)}),
    'peptides-struct': dict(higher=False, rows={
        'GT': (0.2529, 0.0016), 'SAN': (0.2545, 0.0012), 'GraphGPS': (0.2500, 0.0005), 'GraphGPS (2023)': (0.2509, 0.0014),
        'MGT': (0.2453, 0.0025), 'DRew': (0.2536, 0.0015), 'Graph ViT/MLP-Mixer': (0.2449, 0.0016),
        'Exphormer': (0.2512, 0.0025), 'GRIT': (0.2460, 0.0012), 'Subgraphormer': (0.2475, 0.0007),
        'GRED': (0.2455, 0.0013), 'GEAET': (0.2547, 0.0009), 'TIGT': (0.2485, 0.0015), 'GECO': (0.2464, 0.0009),
        'GPNN': (0.2454, 0.0003), 'Graph-Mamba': (0.2478, 0.0016), 'GSSC': (0.2459, 0.0020)}),
    'ogbg-molhiv': dict(higher=True, rows={  # pretrained Graphormer and EGT excluded, as in the paper's ranking
        'SAN': (0.7785, 0.2470), 'GraphGPS': (0.7880, 0.0101), 'Specformer': (0.7889, 0.0124),
        'Graph ViT/MLP-Mixer': (0.7997, 0.0102), 'Exphormer': (0.7834, 0.0044), 'GRIT': (0.7835, 0.0054),
        'Subgraphormer': (0.8038, 0.0192), 'GECO': (0.7980, 0.0200), 'GSSC': (0.8035, 0.0142)}),
}
CLAIMS = [('R1', 'peptides-func', 'gcnplus_peptides-func', 'GCN+ ranks first on Peptides-func'),
          ('R2', 'peptides-struct', 'gcnplus_peptides-struct', 'GCN+ ranks first on Peptides-struct'),
          ('R3', 'ogbg-molhiv', 'gatedgcnplus_ogbg-molhiv', 'GatedGCN+ ranks first on ogbg-molhiv')]


def ours(runs, stem):
    xs = [json.load(open(f))['test_at_best_val'] for f in sorted(glob.glob(os.path.join(runs, stem + '_s*', 'summary.json')))]
    return statistics.mean(xs), statistics.stdev(xs), len(xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='results/runs')
    args = parser.parse_args()
    for claim, ds, stem, text in CLAIMS:
        m, s, n = ours(args.runs, stem)
        spec = COMPETITORS[ds]
        better = (lambda a, b: a > b) if spec['higher'] else (lambda a, b: a < b)
        rank = 1 + sum(better(cm, m) for cm, _ in spec['rows'].values())
        print(f'## {claim}: {text}\n')
        print(f'Ours: {m:.4f} ± {s:.4f} ({n} seeds). Rank by mean among the {len(spec["rows"])} cited competitors: **{rank}**.\n')
        print('| Competitor | Cited | Gap (ours − theirs) | 2·SE (n_comp = 1) | Verdict |')
        print('| --- | --- | --- | --- | --- |')
        ahead = [kv for kv in spec['rows'].items() if better(kv[1][0], m)]
        behind = sorted((kv for kv in spec['rows'].items() if not better(kv[1][0], m)),
                        key=lambda kv: abs(kv[1][0] - m))[:3]
        shown = sorted(ahead + behind, key=lambda kv: kv[1][0], reverse=spec['higher'])
        indist = []
        for name, (cm, cs) in shown:
            gap = m - cm
            se2 = 2 * math.sqrt(s ** 2 / n + cs ** 2)
            if abs(gap) <= se2:
                verdict = 'indistinguishable'
                indist.append(name)
            else:
                verdict = 'ours better' if better(m, cm) else 'theirs better'
            print(f'| {name} | {cm:.4f} ± {cs:.4f} | {gap:+.4f} | {se2:.4f} | {verdict} |')
        print(f'\nIndistinguishable from (among those shown: every competitor ahead of ours plus the three closest behind): {", ".join(indist) or "none"}.\n')


if __name__ == '__main__':
    main()
