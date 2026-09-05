"""
fusion_model.py
================
Task 3 (Hard): GNN-BERT fusion for multi-label tag/genre prediction
plus optional valence/arousal (emotion) regression.

Implements both a simple concatenation fusion and a cross-attention
fusion, selectable via config.yaml -> fusion.type, so the two can be
compared directly in the ablation study.
"""

import torch
import torch.nn as nn

from gnn_model import GraphSAGEEncoder, GATEncoder
from bert_encoder import BertTextEncoder


class CrossAttentionFusion(nn.Module):
    """
    A = softmax(QK^T / sqrt(d)),  Q = g W_Q,  K = H_text W_K
    z = CONCAT(g, A H_text)
    """

    def __init__(self, graph_dim: int, text_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(graph_dim, hidden_dim)
        self.k_proj = nn.Linear(text_dim, hidden_dim)
        self.v_proj = nn.Linear(text_dim, hidden_dim)
        self.num_heads = num_heads
        self.out_dim = graph_dim + hidden_dim

    def forward(self, g, tokens, attention_mask):
        """
        g:              (batch, graph_dim)          -- graph-level embedding
        tokens:         (batch, seq_len, text_dim)   -- full BERT token sequence
        attention_mask: (batch, seq_len)
        """
        Q = self.q_proj(g).unsqueeze(1)          # (batch, 1, hidden_dim)
        K = self.k_proj(tokens)                  # (batch, seq_len, hidden_dim)
        V = self.v_proj(tokens)                  # (batch, seq_len, hidden_dim)

        scale = self.hidden_dim ** 0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale     # (batch, 1, seq_len)

        # mask out padding tokens before softmax
        mask = (attention_mask == 0).unsqueeze(1)             # (batch, 1, seq_len)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)                  # (batch, 1, seq_len)
        attended_text = torch.bmm(attn, V).squeeze(1)          # (batch, hidden_dim)

        z = torch.cat([g, attended_text], dim=-1)              # (batch, graph_dim + hidden_dim)
        return z


class ConcatFusion(nn.Module):
    """Simple baseline fusion: z = CONCAT(g, t). Used in the ablation study."""

    def __init__(self, graph_dim: int, text_dim: int):
        super().__init__()
        self.out_dim = graph_dim + text_dim

    def forward(self, g, t):
        return torch.cat([g, t], dim=-1)


class GNNBertFusionModel(nn.Module):
    """
    End-to-end fusion model:
      - GNN encoder over the music structure graph -> g
      - BERT encoder over tags/captions/lyrics -> t (+ full token sequence for cross-attn)
      - Fusion -> z
      - Multi-label tag head: yhat = sigmoid(W z + b)
      - Optional emotion regression heads: valence, arousal
    """

    def __init__(
        self,
        graph_in_dim: int,
        num_labels: int,
        bert_model_name: str = "distilbert-base-uncased",
        gnn_hidden_dim: int = 128,
        gnn_out_dim: int = 128,
        gnn_layers: int = 3,
        gnn_architecture: str = "graphsage",
        gnn_dropout: float = 0.3,
        fusion_type: str = "cross_attention",
        fusion_hidden_dim: int = 256,
        fusion_heads: int = 4,
        bert_finetune: bool = False,
        predict_emotion: bool = True,
    ):
        super().__init__()

        if gnn_architecture == "gat":
            self.gnn = GATEncoder(graph_in_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout=gnn_dropout)
        else:
            self.gnn = GraphSAGEEncoder(graph_in_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout=gnn_dropout)

        self.bert = BertTextEncoder(model_name=bert_model_name, finetune=bert_finetune)

        self.fusion_type = fusion_type
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(gnn_out_dim, self.bert.hidden_dim, fusion_hidden_dim, fusion_heads)
        else:
            self.fusion = ConcatFusion(gnn_out_dim, self.bert.hidden_dim)

        self.tag_head = nn.Linear(self.fusion.out_dim, num_labels)

        self.predict_emotion = predict_emotion
        if predict_emotion:
            self.valence_head = nn.Linear(self.fusion.out_dim, 1)
            self.arousal_head = nn.Linear(self.fusion.out_dim, 1)

    def forward(self, graph_x, edge_index, graph_batch, input_ids, attention_mask):
        _, g = self.gnn(graph_x, edge_index, graph_batch)                 # (batch, gnn_out_dim)
        t, tokens = self.bert(input_ids, attention_mask)                   # (batch, hidden), (batch, seq, hidden)

        if self.fusion_type == "cross_attention":
            z = self.fusion(g, tokens, attention_mask)
        else:
            z = self.fusion(g, t)

        tag_logits = self.tag_head(z)

        valence, arousal = None, None
        if self.predict_emotion:
            valence = self.valence_head(z).squeeze(-1)
            arousal = self.arousal_head(z).squeeze(-1)

        return {"tag_logits": tag_logits, "valence": valence, "arousal": arousal, "z": z}


def fusion_multitask_loss(outputs, y_tags, y_valence=None, y_arousal=None, alpha=0.5, beta=0.5):
    """
    L = L_tags + alpha * ||v - vhat||^2 + beta * ||a - ahat||^2
    """
    bce = nn.BCEWithLogitsLoss()
    loss = bce(outputs["tag_logits"], y_tags)

    # NaN targets mark tracks without emotion annotations (see datasets.py);
    # mask them out so they don't poison the loss with NaN gradients.
    if y_valence is not None and outputs["valence"] is not None:
        valid = ~torch.isnan(y_valence)
        if valid.any():
            loss = loss + alpha * torch.nn.functional.mse_loss(
                outputs["valence"][valid], y_valence[valid]
            )
    if y_arousal is not None and outputs["arousal"] is not None:
        valid = ~torch.isnan(y_arousal)
        if valid.any():
            loss = loss + beta * torch.nn.functional.mse_loss(
                outputs["arousal"][valid], y_arousal[valid]
            )

    return loss


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    g1 = Data(x=torch.randn(5, 140), edge_index=torch.randint(0, 5, (2, 8)))
    g2 = Data(x=torch.randn(7, 140), edge_index=torch.randint(0, 7, (2, 10)))
    batch = Batch.from_data_list([g1, g2])

    model = GNNBertFusionModel(graph_in_dim=140, num_labels=50, fusion_type="cross_attention")
    texts = ["melancholic piano ballad", "upbeat electronic dance track"]
    enc = model.bert.tokenize(texts)

    out = model(batch.x, batch.edge_index, batch.batch, enc["input_ids"], enc["attention_mask"])
    print(f"tag_logits: {out['tag_logits'].shape}, valence: {out['valence'].shape}")
