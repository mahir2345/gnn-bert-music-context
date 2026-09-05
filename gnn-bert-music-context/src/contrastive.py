"""
contrastive.py
===============
Task 4 (Advanced): contrastive dual-encoder for cross-modal alignment
between audio structure graphs and MusicCaps natural-language captions.

L_NCE = -log( exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau) )
sim(u, v) = u^T v / (||u|| ||v||)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import GraphSAGEEncoder
from bert_encoder import BertTextEncoder


class DualEncoder(nn.Module):
    """
    Two independent encoders projected into a shared embedding space:
      g = Normalize(Proj_g(GNN(G)))
      t = Normalize(Proj_t(BERT_CLS(caption)))
    """

    def __init__(
        self,
        graph_in_dim: int,
        bert_model_name: str = "distilbert-base-uncased",
        gnn_hidden_dim: int = 128,
        gnn_out_dim: int = 128,
        gnn_layers: int = 3,
        embed_dim: int = 256,
        bert_finetune: bool = False,
    ):
        super().__init__()
        self.gnn = GraphSAGEEncoder(graph_in_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers)
        self.bert = BertTextEncoder(model_name=bert_model_name, finetune=bert_finetune)

        self.graph_proj = nn.Linear(gnn_out_dim, embed_dim)
        self.text_proj = nn.Linear(self.bert.hidden_dim, embed_dim)

    def encode_graph(self, x, edge_index, batch):
        _, g = self.gnn(x, edge_index, batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        t, _ = self.bert(input_ids, attention_mask)
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)

    def forward(self, graph_x, edge_index, graph_batch, input_ids, attention_mask):
        g = self.encode_graph(graph_x, edge_index, graph_batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g, t, temperature: float = 0.07):
    """
    Symmetric InfoNCE loss over a batch of N paired (graph, caption) embeddings.
    g, t: (N, embed_dim), both L2-normalized.
    """
    logits = g @ t.t() / temperature           # (N, N) similarity matrix S_ij
    targets = torch.arange(g.size(0), device=g.device)

    loss_g2t = F.cross_entropy(logits, targets)         # caption retrieval from graph
    loss_t2g = F.cross_entropy(logits.t(), targets)      # graph retrieval from caption

    return (loss_g2t + loss_t2g) / 2


@torch.no_grad()
def retrieval_metrics(g, t, ks=(1, 5, 10)):
    """
    Computes Recall@K for both directions: caption->audio (t2g) and audio->caption (g2t).
    g, t: (N, embed_dim), L2-normalized, index-aligned (row i in g pairs with row i in t).
    """
    sim = g @ t.t()   # (N, N); sim[i, j] = similarity of graph i with caption j
    n = sim.size(0)
    results = {}

    for direction, matrix in [("audio_to_caption", sim), ("caption_to_audio", sim.t())]:
        ranks = matrix.argsort(dim=-1, descending=True)   # (N, N) sorted candidate indices
        correct = ranks == torch.arange(n, device=g.device).unsqueeze(1)
        for k in ks:
            hit_at_k = correct[:, :k].any(dim=-1).float().mean().item()
            results[f"{direction}_R@{k}"] = hit_at_k

    return results


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    graphs = [Data(x=torch.randn(5, 140), edge_index=torch.randint(0, 5, (2, 8))) for _ in range(8)]
    batch = Batch.from_data_list(graphs)

    model = DualEncoder(graph_in_dim=140, embed_dim=256)
    texts = [f"caption number {i}" for i in range(8)]
    enc = model.bert.tokenize(texts)

    g, t = model(batch.x, batch.edge_index, batch.batch, enc["input_ids"], enc["attention_mask"])
    loss = info_nce_loss(g, t, temperature=0.07)
    metrics = retrieval_metrics(g, t, ks=(1, 5))

    print(f"g: {g.shape}, t: {t.shape}, loss: {loss.item():.4f}")
    print(f"retrieval metrics (random init, sanity check only): {metrics}")
