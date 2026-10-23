"""Train one config with one seed and log every epoch to CSV.

Usage:
  python -m src.train configs/runs/gcnplus_peptides-func.yaml --seed 0
  python -m src.train <config> --epochs 3 --limit-batches 60 --set train.warmup_epochs=1   # CPU smoke test

An "epoch" is one evaluation period. By default it is one pass over the training
set. With train.steps_per_epoch set (arm A, fixed optimizer steps), it is that
many optimizer steps, cycling through a smaller training subset as needed.
Outputs: <out>/<run>/epochs.csv and <out>/<run>/summary.json.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from src.data.datasets import load_splits, nested_subset, task_info
from src.gnnplus.baselines import OGBGCN, TonshoffGCN
from src.gnnplus.layers import ModelConfig
from src.gnnplus.model import GNNPlus, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, num_tasks):
    arch, params = cfg['arch'], dict(cfg.get('model', {}))
    if arch == 'gnnplus':
        return GNNPlus(ModelConfig(**params, num_tasks=num_tasks))
    if arch == 'tonshoff_gcn':
        return TonshoffGCN(num_tasks=num_tasks, **params)
    if arch == 'ogb_gcn':
        return OGBGCN(num_tasks=num_tasks, **params)
    raise ValueError(f'unknown arch {arch}')


def rwse_steps(cfg):
    return int(cfg.get('model', {}).get('pe_steps', 0)) if cfg['arch'] != 'ogb_gcn' else 0


def load_subset(path, num_graphs):
    """Read a committed training-subset index file and check it against its stored hash."""
    with open(path) as f:
        spec = json.load(f)
    idx = spec['indices']
    digest = hashlib.sha256(','.join(map(str, idx)).encode()).hexdigest()
    if digest != spec['sha256'] or spec['num_train_graphs'] != num_graphs:
        raise ValueError(f'subset file {path} does not match this training split')
    return idx


def loss_fn(kind, pred, y):
    if kind == 'bce':
        mask = ~torch.isnan(y)
        return F.binary_cross_entropy_with_logits(pred[mask], y[mask])
    return F.l1_loss(pred, y)


def metric_fn(kind, pred, y):
    """AP: mean over tasks with both classes present (LRGB). MAE: over all targets.
    ROC-AUC: the OGB evaluator."""
    pred, y = pred.numpy(), y.numpy()
    if kind == 'ap':
        scores = [average_precision_score(y[:, t], pred[:, t]) for t in range(y.shape[1])
                  if 0 < np.nansum(y[:, t]) < np.sum(~np.isnan(y[:, t]))]
        return float(np.mean(scores))
    if kind == 'mae':
        return float(np.abs(pred - y).mean())
    if kind == 'acc':
        return float(((pred > 0) == (y > 0.5)).mean())
    from ogb.graphproppred import Evaluator
    return float(Evaluator('ogbg-molhiv').eval({'y_true': y.astype(int), 'y_pred': pred})['rocauc'])


def cosine_with_warmup(optimizer, warmup, total):
    """GraphGPS-style schedule, stepped once per epoch (A3 blind default)."""
    def factor(epoch):
        if epoch < warmup:
            return max(1e-6, epoch / max(1, warmup))
        progress = (epoch - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def recompute_bn_stats(model, loader, device):
    """'Precise BN': replace BatchNorm running statistics with a cumulative average over one
    no-gradient pass of the training set. Used where running averages lag a fast-changing model
    on near-constant features (the synthetic task's constant node features); off by default."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = [b.momentum for b in bns]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.train()
    for batch in loader:
        model(batch.to(device))
    for b, m in zip(bns, momenta):
        b.momentum = m


@torch.no_grad()
def evaluate(model, loader, task, device, limit=None):
    model.eval()
    preds, ys, losses = [], [], []
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        batch = batch.to(device)
        pred = model(batch)
        losses.append(loss_fn(task['loss'], pred, batch.y).item() * batch.num_graphs)
        preds.append(pred.cpu())
        ys.append(batch.y.cpu())
    pred, y = torch.cat(preds), torch.cat(ys)
    return sum(losses) / len(y), metric_fn(task['metric'], pred, y)


def endless(loader):
    while True:
        yield from loader


def run(cfg, seed, out_dir, run_name, epochs=None, limit_batches=None, data_root='data', device=None):
    run_start = time.time()
    task = task_info(cfg['dataset'])
    tr = dict(cfg['train'])
    for key in ('lr', 'weight_decay', 'clip_grad_norm'):  # YAML reads '1e-05' as a string
        if key in tr:
            tr[key] = float(tr[key])
    epochs = epochs or tr['epochs']
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    splits = load_splits(cfg['dataset'], data_root, rwse_steps=rwse_steps(cfg))
    train_graphs = splits['train']
    if tr.get('subset_file'):  # arm A: committed index file, checked by hash
        idx = load_subset(tr['subset_file'], len(train_graphs))
        train_graphs = [train_graphs[i] for i in idx]
    elif tr.get('train_fraction', 1.0) < 1.0:
        idx = nested_subset(len(train_graphs), tr['train_fraction'], tr.get('subset_seed', 0))
        train_graphs = [train_graphs[i] for i in idx.tolist()]

    set_seed(seed)
    model = build_model(cfg, task['num_tasks']).to(device)
    n_params = count_parameters(model)
    expected = cfg.get('expected_params')
    if expected is not None and n_params != expected:
        raise ValueError(f'{n_params} parameters, config expects {expected}')

    loader_gen = torch.Generator().manual_seed(seed)
    workers = int(tr.get('num_workers', 0))
    extra = dict(num_workers=workers, persistent_workers=True) if workers else {}
    train_loader = DataLoader(train_graphs, batch_size=tr['batch_size'], shuffle=True, generator=loader_gen, **extra)
    bn_stats_loader = DataLoader(train_graphs, batch_size=tr['batch_size']) if tr.get('precise_bn') else None
    eval_bs = max(tr['batch_size'], 128)
    val_loader = DataLoader(splits['val'], batch_size=eval_bs, **extra)
    test_loader = DataLoader(splits['test'], batch_size=eval_bs, **extra)

    if tr.get('optimizer', 'adamw') == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=tr['lr'], weight_decay=tr.get('weight_decay', 0.0))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr.get('weight_decay', 0.0))
    scheduler = None
    if tr.get('scheduler', 'none') == 'cosine_with_warmup':
        scheduler = cosine_with_warmup(optimizer, tr['warmup_epochs'], tr['epochs'])

    steps_per_epoch = tr.get('steps_per_epoch') or len(train_loader)
    if limit_batches:
        steps_per_epoch = min(steps_per_epoch, limit_batches)
    batches = endless(train_loader)
    clip = tr.get('clip_grad_norm', 0.0)

    os.makedirs(os.path.join(out_dir, run_name), exist_ok=True)
    csv_path = os.path.join(out_dir, run_name, 'epochs.csv')
    fields = ['epoch', 'lr', 'train_loss', 'train_metric', 'val_loss', 'val_metric', 'test_loss',
              'test_metric', 'seconds', 'end_time']
    rows = []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        setup_seconds = time.time() - run_start
        for epoch in range(epochs):
            start = time.time()
            model.train()
            preds, ys, losses = [], [], []
            lr = optimizer.param_groups[0]['lr']
            for _ in range(steps_per_epoch):
                batch = next(batches).to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = loss_fn(task['loss'], pred, batch.y)
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                losses.append(loss.item() * batch.num_graphs)
                preds.append(pred.detach().cpu())
                ys.append(batch.y.cpu())
            y_train = torch.cat(ys)
            train_metric = metric_fn(task['metric'], torch.cat(preds), y_train)
            if bn_stats_loader is not None:
                recompute_bn_stats(model, bn_stats_loader, device)
            val_loss, val_metric = evaluate(model, val_loader, task, device, limit_batches)
            test_loss, test_metric = evaluate(model, test_loader, task, device, limit_batches)
            if scheduler is not None:
                scheduler.step()
            row = dict(epoch=epoch, lr=lr, train_loss=sum(losses) / len(y_train), train_metric=train_metric,
                       val_loss=val_loss, val_metric=val_metric, test_loss=test_loss, test_metric=test_metric,
                       seconds=time.time() - start, end_time=time.time())
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f'[{run_name}] epoch {epoch}: train_loss {row["train_loss"]:.4f} '
                  f'val {task["metric"]} {val_metric:.4f} test {test_metric:.4f} ({row["seconds"]:.1f}s)', flush=True)

    pick = max if task['higher_is_better'] else min
    best = pick(rows, key=lambda r: r['val_metric'])
    summary = dict(run=run_name, dataset=cfg['dataset'], arch=cfg['arch'], seed=seed, params=n_params,
                   train_graphs=len(train_graphs), steps_per_epoch=steps_per_epoch,
                   epochs_run=len(rows), best_epoch=best['epoch'], best_val=best['val_metric'],
                   test_at_best_val=best['test_metric'], test_at_last_epoch=rows[-1]['test_metric'],
                   mean_seconds_per_epoch=float(np.mean([r['seconds'] for r in rows])),
                   setup_seconds=setup_seconds, wall_seconds=time.time() - run_start,
                   device=str(device), torch=torch.__version__, smoke_test=bool(limit_batches))
    with open(os.path.join(out_dir, run_name, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, help='override train.epochs (schedule still uses train.epochs)')
    parser.add_argument('--limit-batches', type=int, help='cap train steps and eval batches per epoch (smoke tests)')
    parser.add_argument('--out', default='results/runs')
    parser.add_argument('--tag', default='')
    parser.add_argument('--device')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override config entries, e.g. train.warmup_epochs=1 model.pe_steps=0')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        key, value = item.split('=', 1)
        *parents, leaf = key.split('.')
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
        if key.startswith('model.'):
            cfg.pop('expected_params', None)  # a changed architecture no longer has the reported count
    name = os.path.splitext(os.path.basename(args.config))[0] + f'_s{args.seed}' + (f'_{args.tag}' if args.tag else '')
    summary = run(cfg, args.seed, args.out, name, args.epochs, args.limit_batches, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
