"""GNN+ model: input encoders (Eq. 12), stacked GNN+ layers, readout (Eq. 5), head."""
import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from src.gnnplus.layers import ACTIVATIONS, GNNPlusLayer, ModelConfig

POOLING = {'mean': global_mean_pool, 'add': global_add_pool, 'sum': global_add_pool, 'max': global_max_pool}


class NodeEncoder(nn.Module):
    """Atom embedding plus RWSE (Eq. 12: x_v = [x_v || x_v^RWSE] W_PE), per A7.

    'concat_project': embed atoms to d, concatenate raw RWSE, one linear map to d.
    'gps_concat': atoms to d - pe_dim, RWSE linearly to pe_dim, concatenate (GraphGPS).
    'add': atoms to d, plus a linear map of RWSE to d. This equals Eq. 12 applied
           to one-hot atom features, so it is also the "x_v is raw" reading.
    """

    def __init__(self, cfg):
        super().__init__()
        d, k = cfg.hidden, cfg.pe_steps
        self.mode = cfg.pe_mode if k else None
        self.pe_norm = nn.BatchNorm1d(k) if k and cfg.pe_norm else None
        if self.mode == 'gps_concat':
            self.atom = AtomEncoder(d - cfg.pe_dim)
            self.pe_lin = nn.Linear(k, cfg.pe_dim)
        else:
            self.atom = AtomEncoder(d)
            if self.mode == 'concat_project':
                self.proj = nn.Linear(d + k, d)
            elif self.mode == 'add':
                self.pe_lin = nn.Linear(k, d)

    def forward(self, x, rwse):
        h = self.atom(x)
        if self.mode is None:
            return h
        if self.pe_norm is not None:
            rwse = self.pe_norm(rwse)
        if self.mode == 'concat_project':
            return self.proj(torch.cat([h, rwse], dim=1))
        if self.mode == 'gps_concat':
            return torch.cat([h, self.pe_lin(rwse)], dim=1)
        return h + self.pe_lin(rwse)


def build_head(d, out, kind, n_linear, act, dropout=0.0):
    """Prediction head g(.) (A10). n_linear counts every Linear, including the output."""
    layers = []
    width = d
    for _ in range(n_linear - 1):
        nxt = width if kind == 'mlp' else width // 2
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(width, nxt), act()]
        width = nxt
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(width, out))
    return nn.Sequential(*layers)


class GNNPlus(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.node_enc = NodeEncoder(cfg)
        shared_edges = cfg.bond_encoder_always or (
            cfg.edge and (cfg.conv == 'gatedgcn' or cfg.edge_encoder in ('shared_linear', 'shared', 'shared_pre_linear')))
        self.edge_enc = BondEncoder(d) if shared_edges else None
        self.layers = nn.ModuleList(GNNPlusLayer(cfg) for _ in range(cfg.layers))
        self.pool = POOLING[cfg.pooling]
        self.head = build_head(d, cfg.num_tasks, cfg.head, cfg.head_layers, ACTIVATIONS[cfg.act],
                               cfg.dropout if cfg.head_dropout else 0.0)

    def forward(self, batch):
        x = self.node_enc(batch.x, getattr(batch, 'rwse', None))
        e = self.edge_enc(batch.edge_attr) if self.edge_enc is not None else None
        for layer in self.layers:
            x, e = layer(x, batch.edge_index, batch.edge_attr, e)
        return self.head(self.pool(x, batch.batch))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
