"""Parameter-count matching (AMBIGUITIES.md, resolution method).

Tables 9 and 13 report an exact parameter count for each tuned config. Every
combination of readings predicts an exact count. This script enumerates the
readings, counts analytically (checked against instantiated models), and lists
the combinations that hit all three in-scope counts per model.

Architecture-level readings (encoders, conv internals) must be shared across
datasets, since the code is shared. The prediction head and the RWSE width
(pe_dim) are per-config settings in GraphGPS-style code, so they may differ
per dataset.

Usage: python -m src.param_count [--out results/param_count_matching.md]
"""
import argparse
import itertools
import random
from dataclasses import replace

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from src.gnnplus.layers import ModelConfig
from src.gnnplus.model import GNNPlus, count_parameters

ATOM_VOCAB = sum(get_atom_feature_dims())  # 174 in ogb 1.3.6; older releases had 173
BOND_VOCAB = sum(get_bond_feature_dims())  # 13

# The six in-scope tuned configs (configs/paper/in_scope.yaml) and their reported counts.
TARGETS = {
    'gcn': {
        'peptides-func': (dict(layers=3, hidden=275, edge=True, residual=False, ffn=False, pe_steps=32, num_tasks=10), 507351),
        'peptides-struct': (dict(layers=5, hidden=255, edge=False, residual=False, ffn=False, pe_steps=32, num_tasks=11), 506127),
        'ogbg-molhiv': (dict(layers=4, hidden=256, edge=True, residual=False, ffn=True, pe_steps=20, num_tasks=1), 1407641),
    },
    'gatedgcn': {
        'peptides-func': (dict(layers=5, hidden=135, edge=True, residual=False, ffn=False, pe_steps=32, num_tasks=10), 521141),
        'peptides-struct': (dict(layers=4, hidden=145, edge=True, residual=True, ffn=False, pe_steps=32, num_tasks=11), 492897),
        'ogbg-molhiv': (dict(layers=3, hidden=256, edge=True, residual=True, ffn=False, pe_steps=20, num_tasks=1), 1076633),
    },
}

# Architecture-level readings, shared across datasets.
ARCH_OPTIONS = {
    'gcn': dict(
        pe_mode=['concat_project', 'add', 'gps_concat'],
        pe_norm=[True, False],
        edge_encoder=['shared_linear', 'per_layer_bond', 'shared'],  # 'shared_pre_linear' counts like 'shared'
        bond_encoder_always=[True, False],
        edge_bias=[True, False],
        self_loop_edge=['zero', 'learned'],
        conv_bias=[True, False],
        ffn_ratio=[1, 2],
        atom_vocab=[ATOM_VOCAB, ATOM_VOCAB - 1],
    ),
    'gatedgcn': dict(
        pe_mode=['concat_project', 'add', 'gps_concat'],
        pe_norm=[True, False],
        gated_variant=['dwivedi', 'dwivedi_noedgeupdate'],  # 'eq4_msg' has the same count as the latter
        ffn_ratio=[1, 2],
        atom_vocab=[ATOM_VOCAB, ATOM_VOCAB - 1],
    ),
}
# Per-config settings.
HEADS = [('mlp', 1), ('mlp', 2), ('mlp', 3), ('mlp', 4), ('halving', 2), ('halving', 3), ('halving', 4)]
PE_DIMS = [8, 12, 16, 20, 24, 28, 32]


def analytic_count(cfg: ModelConfig, atom_vocab=ATOM_VOCAB):
    d, k, t = cfg.hidden, cfg.pe_steps, cfg.num_tasks
    n = 0
    # Node encoder
    if not k:
        n += atom_vocab * d
    else:
        n += 2 * k if cfg.pe_norm else 0
        if cfg.pe_mode == 'concat_project':
            n += atom_vocab * d + (d + k) * d + d
        elif cfg.pe_mode == 'add':
            n += atom_vocab * d + k * d + d
        else:
            n += atom_vocab * (d - cfg.pe_dim) + k * cfg.pe_dim + cfg.pe_dim
    # Shared edge encoder
    if cfg.bond_encoder_always or (
            cfg.edge and (cfg.conv == 'gatedgcn' or cfg.edge_encoder in ('shared_linear', 'shared', 'shared_pre_linear'))):
        n += BOND_VOCAB * d
    # Layers
    per_layer = 2 * d if cfg.bn else 0
    if cfg.conv == 'gcn':
        per_layer += d * d + (d if cfg.conv_bias else 0)
        if cfg.edge:
            if cfg.edge_encoder == 'shared_linear':
                per_layer += d * d + (d if cfg.edge_bias else 0)
            elif cfg.edge_encoder == 'per_layer_bond':
                per_layer += BOND_VOCAB * d
            if cfg.self_loop_edge == 'learned':
                per_layer += d
    else:
        per_layer += 4 * (d * d + d) + ((d * d + d) if cfg.edge else 0)
        if cfg.edge and cfg.gated_variant == 'dwivedi' and cfg.bn:
            per_layer += 2 * d
    if cfg.ffn:
        r = cfg.ffn_ratio * d
        per_layer += d * r + r + r * d + d + 2 * d + (2 * d if cfg.ffn_style == 'code' else 0)
    n += cfg.layers * per_layer
    # Head
    width = d
    for _ in range(cfg.head_layers - 1):
        nxt = width if cfg.head == 'mlp' else width // 2
        n += width * nxt + nxt
        width = nxt
    n += width * t + t
    return n


def _config(model, base, arch, head, pe_dim):
    fields = {k: v for k, v in arch.items() if k != 'atom_vocab'}
    return ModelConfig(conv=model, **base, **fields, head=head[0], head_layers=head[1], pe_dim=pe_dim)


def self_check(samples=40, seed=0):
    """Analytic counts must equal instantiated counts (current ogb vocab only)."""
    rng = random.Random(seed)
    for _ in range(samples):
        model = rng.choice(['gcn', 'gatedgcn'])
        base, _ = rng.choice(list(TARGETS[model].values()))
        arch = {k: rng.choice(v) for k, v in ARCH_OPTIONS[model].items()}
        arch['atom_vocab'] = ATOM_VOCAB
        cfg = _config(model, base, arch, rng.choice(HEADS), rng.choice(PE_DIMS))
        built = count_parameters(GNNPlus(cfg))
        assert analytic_count(cfg) == built, (cfg, analytic_count(cfg), built)
    return samples


def search(model):
    """All architecture readings for which every dataset has some head/pe_dim hitting its count."""
    keys = list(ARCH_OPTIONS[model])
    matches, near = [], []
    for values in itertools.product(*(ARCH_OPTIONS[model][k] for k in keys)):
        arch = dict(zip(keys, values))
        if model == 'gcn' and arch['edge_encoder'] != 'shared_linear' and not arch['edge_bias']:
            continue  # edge_bias only exists for shared_linear
        per_dataset, best_gap = {}, 0
        for ds, (base, target) in TARGETS[model].items():
            pe_dims = PE_DIMS if arch['pe_mode'] == 'gps_concat' else [0]
            hits, gaps = [], []
            for head, pe_dim in itertools.product(HEADS, pe_dims):
                cfg = _config(model, base, arch, head, pe_dim)
                gap = analytic_count(cfg, arch['atom_vocab']) - target
                gaps.append((abs(gap), gap, head, pe_dim))
                if gap == 0:
                    hits.append((head, pe_dim))
            per_dataset[ds] = hits
            best_gap = max(best_gap, min(gaps)[0])
        if all(per_dataset.values()):
            matches.append((arch, per_dataset))
        else:
            near.append((best_gap, arch))
    near.sort(key=lambda x: x[0])
    return matches, near[:5]


def _fmt_arch(arch):
    return ', '.join(f'{k}={v}' for k, v in arch.items())


def _fmt_hits(hits):
    return '; '.join(f'{h[0]}-{h[1]}' + (f', pe_dim={p}' if p else '') for h, p in hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/param_count_matching.md')
    args = parser.parse_args()
    n = self_check()
    lines = ['# Parameter-count matching', '',
             f'Generated by `python -m src.param_count`. Analytic counts checked against {n} instantiated models.',
             f'Atom vocabulary in installed ogb: {ATOM_VOCAB}; {ATOM_VOCAB - 1} is the older release.', '']
    for model in ('gcn', 'gatedgcn'):
        matches, near = search(model)
        lines += [f'## {model}', '', f'{len(matches)} architecture readings match all three reported counts.', '']
        if matches:
            lines += ['| Architecture reading | ' + ' | '.join(TARGETS[model]) + ' |',
                      '| --- |' + ' --- |' * len(TARGETS[model])]
            for arch, per_dataset in matches:
                lines.append(f'| {_fmt_arch(arch)} | ' + ' | '.join(_fmt_hits(per_dataset[d]) for d in TARGETS[model]) + ' |')
        else:
            lines += ['Closest readings (largest per-dataset gap among the best head choices):', '']
            for gap, arch in near:
                lines.append(f'- off by up to {gap}: {_fmt_arch(arch)}')
        lines.append('')
    text = '\n'.join(lines)
    with open(args.out, 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
