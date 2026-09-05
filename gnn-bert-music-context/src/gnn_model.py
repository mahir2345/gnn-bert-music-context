"""
gnn_model.py
============
GraphSAGE / GAT encoders on music structure graphs (Task 2), plus a
genre/tag classification head and a CNN baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    """
    L-layer GraphSAGE encoder.

    h_i^(l+1) = sigma( W^(l) . CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)) )
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)   # graph-level readout: g = mean_i h_i^(L)
        return x, g                       # x = node embeddings, g = graph embedding


class GATEncoder(nn.Module):
    """Alternative encoder using Graph Attention layers instead of GraphSAGE."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return x, g


class GNNTagClassifier(nn.Module):
    """
    Task 2 (Medium): GNN on segment/chord graphs -> genre / top-tag prediction.

    yhat = sigmoid(W g + b), where g is the mean-pooled graph embedding.
    """

    def __init__(self, in_dim: int, num_labels: int, hidden_dim: int = 128,
                 out_dim: int = 128, num_layers: int = 3, dropout: float = 0.3,
                 architecture: str = "graphsage", gat_heads: int = 4):
        super().__init__()
        if architecture == "gat":
            self.encoder = GATEncoder(in_dim, hidden_dim, out_dim, num_layers, gat_heads, dropout)
        else:
            self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.classifier = nn.Linear(out_dim, num_labels)

    def forward(self, x, edge_index, batch):
        _, g = self.encoder(x, edge_index, batch)
        logits = self.classifier(g)
        return logits, g


class CNNBaseline(nn.Module):
    """
    Baseline B2: simple CNN directly on mel-spectrograms, no graph, no text.
    Used to demonstrate that graph structure adds value over raw local patterns.
    """

    def __init__(self, num_labels: int, n_mels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_labels),
        )

    def forward(self, mel_spec):
        # mel_spec: (batch, 1, n_mels, n_frames)
        feats = self.conv(mel_spec)
        return self.classifier(feats)


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    g1 = Data(x=torch.randn(5, 140), edge_index=torch.randint(0, 5, (2, 8)))
    g2 = Data(x=torch.randn(7, 140), edge_index=torch.randint(0, 7, (2, 10)))
    batch = Batch.from_data_list([g1, g2])

    model = GNNTagClassifier(in_dim=140, num_labels=50)
    logits, g = model(batch.x, batch.edge_index, batch.batch)
    print(f"logits shape: {logits.shape}, graph embedding shape: {g.shape}")
