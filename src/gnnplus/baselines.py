"""The two baselines the paper cites, rebuilt from their own published configs.

TonshoffGCN: the Peptides-func GCN behind GNN+ Table 3's 0.6860 (Tönshoff et al.
2023, configs/LRGB-tuned/peptides-func-GCN.yaml in their repository). Claim C0.
OGBGCN: the ogbg-molhiv GCN behind GNN+ Table 4's 0.7606 (Hu et al. 2020,
examples/graphproppred/mol in the OGB repository). Baseline check C0b.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import degree

from src.gnnplus.model import build_head


class TonshoffGCN(nn.Module):
    """Atom+RWSE encoder (RWSE steps 1..20, BatchNorm, Linear to 28, concatenated to
    a 207-wide atom embedding), 6 x [GCNConv -> GELU -> Dropout(0.1)] with residuals,
    mean pooling, 3-layer MLP head with dropout before each Linear. No BatchNorm in
    the layers. The bond encoder exists but GCNConv never reads it; it is kept so
    the parameter count matches the reported 486k.
    """

    def __init__(self, num_tasks=10, hidden=235, layers=6, dropout=0.1, pe_steps=20, pe_dim=28,
                 head_layers=3):
        super().__init__()
        self.dropout = dropout
        self.atom = AtomEncoder(hidden - pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_steps)
        self.pe_lin = nn.Linear(pe_steps, pe_dim)
        self.unused_bond = BondEncoder(hidden)
        self.convs = nn.ModuleList(GCNConv(hidden, hidden, bias=True) for _ in range(layers))
        self.head = build_head(hidden, num_tasks, 'mlp', head_layers, nn.GELU, dropout)

    def forward(self, batch):
        x = torch.cat([self.atom(batch.x), self.pe_lin(self.pe_norm(batch.rwse))], dim=1)
        for conv in self.convs:
            x = x + F.dropout(F.gelu(conv(x, batch.edge_index)), self.dropout, self.training)
        return self.head(global_mean_pool(x, batch.batch))


class OGBGCNConv(MessagePassing):
    """OGB's molecular GCN layer: bond embeddings inside the message, plus a root term."""

    def __init__(self, dim):
        super().__init__(aggr='add')
        self.linear = nn.Linear(dim, dim)
        self.root_emb = nn.Embedding(1, dim)
        self.bond_encoder = BondEncoder(dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
        return out + F.relu(x + self.root_emb.weight) / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class OGBGCN(nn.Module):
    """5 x [conv -> BN -> ReLU (not after the last) -> Dropout(0.5)], JK=last, mean pool, Linear."""

    def __init__(self, num_tasks=1, hidden=300, layers=5, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.atom = AtomEncoder(hidden)
        self.convs = nn.ModuleList(OGBGCNConv(hidden) for _ in range(layers))
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, batch):
        h = self.atom(batch.x)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, batch.edge_index, batch.edge_attr))
            if i < len(self.convs) - 1:
                h = F.relu(h)
            h = F.dropout(h, self.dropout, self.training)
        return self.head(global_mean_pool(h, batch.batch))
