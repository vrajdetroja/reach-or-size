"""GNN+ building blocks, written from Eqs. 4 and 6-11 of arXiv 2502.09263v3.

Where the paper admits more than one reading, the choice is a ModelConfig
field named after its AMBIGUITIES.md row, and the default is the blind reading.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder

ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU}


@dataclass
class ModelConfig:
    """Defaults are the readings that match all six reported parameter counts
    (results/param_count_matching.md). The pre-count blind defaults are listed
    in AMBIGUITIES.md."""
    conv: str = 'gcn'              # 'gcn' (Eq. 2/6) | 'gatedgcn' (Eq. 4)
    layers: int = 3
    hidden: int = 275
    edge: bool = True              # edge feature module (Eq. 6)
    bn: bool = True                # Eq. 7
    dropout: float = 0.2           # Eq. 8
    residual: bool = False         # Eq. 9
    ffn: bool = False              # Eqs. 10-11
    pe_steps: int = 0              # RWSE-K (Eq. 12); 0 = no PE
    pooling: str = 'mean'
    num_tasks: int = 10
    act: str = 'relu'              # A13
    head: str = 'mlp'              # A10: 'mlp' (d -> d ... -> out) | 'halving' (d -> d/2 -> ... -> out)
    head_layers: int = 1           # A10: number of Linear layers in the head; set per config
    head_dropout: bool = False     # A16
    pe_mode: str = 'gps_concat'    # A7: 'gps_concat' | 'concat_project' | 'add'
    pe_dim: int = 28               # A7: RWSE width for 'gps_concat'; set per config
    pe_norm: bool = True           # A7: BatchNorm on raw RWSE
    edge_encoder: str = 'shared'   # A1/A11: 'shared' | 'shared_pre_linear' | 'shared_linear' | 'per_layer_bond'
    bond_encoder_always: bool = True  # A11: build the shared bond encoder even with the edge module off
    edge_bias: bool = True         # A1: bias on W_e ('shared_linear' only)
    edge_norm: bool = False        # A2: degree-normalize the edge term
    self_loop_edge: str = 'zero'   # A1: 'zero' | 'none' | 'learned'
    conv_bias: bool = True         # bias after GCN aggregation
    gated_variant: str = 'dwivedi'  # A8: 'dwivedi' | 'dwivedi_noedgeupdate' | 'eq4_msg'
    gate_norm: bool = True         # A8: normalize gates over neighbors
    edge_residual: str = 'follow'  # A8: 'follow' the node residual flag | 'always'
    ffn_ratio: int = 2             # A12


class GCNPlusConv(nn.Module):
    """Eq. 6, reading A1(a): the edge term is added per message, inside the sum.

    h_v <- sum_{u in N(v) + {v}} [ h_u W / sqrt(d_u d_v) + e_uv W_e ] (+ bias)

    With the count-matched default edge_encoder='shared', there is no per-layer
    W_e: e_uv is the bond embedding shared by all layers, and self-loops get 0.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden
        self.cfg = cfg
        self.lin = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.zeros(d)) if cfg.conv_bias else None
        if cfg.edge:
            if cfg.edge_encoder == 'shared_linear':
                self.edge_lin = nn.Linear(d, d, bias=cfg.edge_bias)
            elif cfg.edge_encoder == 'per_layer_bond':
                self.edge_enc = BondEncoder(d)
            if cfg.self_loop_edge == 'learned':
                self.self_edge = nn.Parameter(torch.zeros(d))

    def _edge_terms(self, edge_attr, edge_emb, n, device):
        cfg = self.cfg
        if cfg.edge_encoder == 'shared_linear':
            real = self.edge_lin(edge_emb)
        elif cfg.edge_encoder == 'per_layer_bond':
            real = self.edge_enc(edge_attr)
        else:
            real = edge_emb
        d = real.size(1)
        if cfg.self_loop_edge == 'learned':
            loop = self.self_edge
        elif cfg.self_loop_edge == 'zero' and cfg.edge_encoder == 'shared_linear' and self.edge_lin.bias is not None:
            loop = self.edge_lin.bias  # W_e applied to a zero edge feature
        else:
            loop = torch.zeros(d, device=device)
        return torch.cat([real, loop.expand(n, d)])

    def forward(self, x, edge_index, edge_attr, edge_emb):
        n = x.size(0)
        loops = torch.arange(n, device=x.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        deg = torch.zeros(n, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        inv_sqrt = deg.pow(-0.5)
        norm = (inv_sqrt[src] * inv_sqrt[dst]).unsqueeze(1)
        if self.cfg.edge and self.cfg.edge_encoder == 'shared_pre_linear':
            # Edge embedding added to the neighbor state before the shared W: (h_u + e_uv) W.
            edge = self._edge_terms(edge_attr, edge_emb, n, x.device)
            msg = norm * self.lin(x[src] + edge)
        else:
            msg = norm * self.lin(x)[src]
        if self.cfg.edge and self.cfg.edge_encoder != 'shared_pre_linear':
            edge = self._edge_terms(edge_attr, edge_emb, n, x.device)
            if self.cfg.edge_norm:
                edge = norm * edge
            msg = msg + edge
        out = torch.zeros_like(x).index_add_(0, dst, msg)
        return out + self.bias if self.bias is not None else out


class GatedGCNConv(nn.Module):
    """Eq. 4 with an edge feature module (A8).

    'dwivedi': gate from (h_v, h_u, e_uv), gates normalized over neighbors,
    edge states updated each layer (Dwivedi et al. 2023 / Bresson & Laurent).
    'dwivedi_noedgeupdate': the same, but edge states stay at their input value.
    'eq4_msg': Eq. 4 as typeset, plus an additive e_uv W_e message as in Eq. 6.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden
        self.cfg = cfg
        self.A = nn.Linear(d, d)  # W1: self term
        self.B = nn.Linear(d, d)  # W2: neighbor values
        self.D = nn.Linear(d, d)  # W3: gate, receiving node
        self.E = nn.Linear(d, d)  # W4: gate, sending node
        self.C = nn.Linear(d, d) if cfg.edge else None  # edge into gate, or W_e for 'eq4_msg'

    def forward(self, x, edge_index, edge_emb):
        src, dst = edge_index
        ax, bx, dx, ex = self.A(x), self.B(x), self.D(x), self.E(x)
        if self.cfg.gated_variant == 'eq4_msg':
            gate = torch.sigmoid(dx[dst] + ex[src])
            msg = gate * bx[src]
            if self.C is not None:
                msg = msg + self.C(edge_emb)
            return ax + torch.zeros_like(x).index_add_(0, dst, msg), None
        e_hat = dx[dst] + ex[src]
        if self.C is not None:
            e_hat = e_hat + self.C(edge_emb)
        gate = torch.sigmoid(e_hat)
        num = torch.zeros_like(x).index_add_(0, dst, gate * bx[src])
        if self.cfg.gate_norm:
            den = torch.zeros_like(x).index_add_(0, dst, gate)
            num = num / (den + 1e-6)
        edge_out = e_hat if self.cfg.gated_variant == 'dwivedi' and self.C is not None else None
        return ax + num, edge_out


class FFN(nn.Module):
    """Eq. 10: FFN(h) = BN(act(h W1) W2 + h). Note the skip connection inside."""

    def __init__(self, d, ratio, act):
        super().__init__()
        self.lin1 = nn.Linear(d, ratio * d)
        self.lin2 = nn.Linear(ratio * d, d)
        self.act = act()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, h):
        return self.bn(self.lin2(self.act(self.lin1(h))) + h)


class GNNPlusLayer(nn.Module):
    """Eq. 11: h <- FFN(Dropout(act(BN(conv(h)))) + h), each part switchable."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        act = ACTIVATIONS[cfg.act]
        self.conv = GCNPlusConv(cfg) if cfg.conv == 'gcn' else GatedGCNConv(cfg)
        self.bn = nn.BatchNorm1d(d) if cfg.bn else None
        self.act = act()
        self.ffn = FFN(d, cfg.ffn_ratio, act) if cfg.ffn else None
        updates_edges = cfg.conv == 'gatedgcn' and cfg.edge and cfg.gated_variant == 'dwivedi'
        self.edge_bn = nn.BatchNorm1d(d) if updates_edges and cfg.bn else None
        self.edge_act = act() if updates_edges else None

    def forward(self, x, edge_index, edge_attr, edge_emb):
        cfg = self.cfg
        if cfg.conv == 'gcn':
            h, e_new = self.conv(x, edge_index, edge_attr, edge_emb), None
        else:
            h, e_new = self.conv(x, edge_index, edge_emb)
        if self.bn is not None:
            h = self.bn(h)
        h = F.dropout(self.act(h), cfg.dropout, self.training)
        if cfg.residual:
            h = h + x
        if self.ffn is not None:
            h = self.ffn(h)
        if e_new is not None:
            if self.edge_bn is not None:
                e_new = self.edge_bn(e_new)
            e_new = F.dropout(self.edge_act(e_new), cfg.dropout, self.training)
            if cfg.residual or cfg.edge_residual == 'always':
                e_new = e_new + edge_emb
            edge_emb = e_new
        return h, edge_emb
