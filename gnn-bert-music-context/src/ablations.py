"""
ablations.py
============
Task 3 ablation study (spec Section 4.3):

  1. BERT-only        -> Task 1 model   (results/metrics/task1_history.json)
  2. GNN-only         -> Task 2 model   (results/metrics/task2_history.json)
  3. Early concat     -> trained HERE   (fusion.type = concat)
  4. Cross-attention  -> Task 3 model   (results/metrics/task3_history.json)

This script trains the concat-fusion variant (identical settings to Task 3
except fusion.type) and then writes a combined summary of the best
validation metrics for all four ablation rows to
results/metrics/ablations.json.

Usage (from repo root):
    python src/ablations.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

from bert_encoder import BertTextEncoder
from datasets import MusicGraphCaptionDataset
from evaluate import compute_multilabel_metrics, find_best_threshold
from fusion_model import GNNBertFusionModel, fusion_multitask_loss


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    want_cuda = cfg["device"] == "cuda"
    return torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")


@torch.no_grad()
def evaluate_fusion(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        graph_batch, input_ids, attention_mask, y_tags = [
            v.to(device) if hasattr(v, "to") else v for v in batch[:4]
        ]
        out = model(
            graph_batch.x, graph_batch.edge_index, graph_batch.batch,
            input_ids, attention_mask,
        )
        all_probs.append(torch.sigmoid(out["tag_logits"]).cpu().numpy())
        all_labels.append(y_tags.cpu().numpy())
    if not all_probs:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "macro_auc_pr": float("nan")}
    y, p = np.concatenate(all_labels), np.concatenate(all_probs)
    best_t, _ = find_best_threshold(y, p)
    metrics = compute_multilabel_metrics(y, p, threshold=best_t)
    metrics["threshold"] = float(best_t)
    return metrics


def train_concat_fusion(cfg):
    """Same training recipe as train.py::train_task3, but fusion.type=concat
    and separate checkpoint/history paths so Task 3 artifacts are untouched."""
    cfg = copy.deepcopy(cfg)
    cfg["fusion"]["type"] = "concat"
    device = get_device(cfg)

    tokenizer = BertTextEncoder(
        model_name=cfg["text"]["model_name"], finetune=False,
        max_length=cfg["text"]["max_length"],
    ).tokenizer

    train_ds = MusicGraphCaptionDataset(split="train", cfg=cfg, tokenizer=tokenizer, include_emotion=True)
    val_ds = MusicGraphCaptionDataset(split="val", cfg=cfg, tokenizer=tokenizer, include_emotion=True)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=val_ds.collate_fn)

    model = GNNBertFusionModel(
        graph_in_dim=train_ds.node_feature_dim,
        num_labels=train_ds.num_labels,
        bert_model_name=cfg["text"]["model_name"],
        gnn_hidden_dim=cfg["gnn"]["hidden_dim"],
        gnn_out_dim=cfg["gnn"]["out_dim"],
        gnn_layers=cfg["gnn"]["num_layers"],
        gnn_architecture=cfg["gnn"]["architecture"],
        gnn_dropout=cfg["gnn"]["dropout"],
        fusion_type="concat",
        fusion_hidden_dim=cfg["fusion"]["hidden_dim"],
        fusion_heads=cfg["fusion"]["num_heads"],
        bert_finetune=cfg["train"]["bert_finetune"],
        predict_emotion=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    alpha = cfg["fusion"]["emotion_loss_weight_valence"]
    beta = cfg["fusion"]["emotion_loss_weight_arousal"]
    patience = cfg["train"].get("early_stopping_patience", 5)
    best_f1, bad_epochs = -1.0, 0
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Ablation:concat] Epoch {epoch+1}"):
            graph_batch, input_ids, attention_mask, y_tags, y_val, y_aro = [
                v.to(device) if hasattr(v, "to") else v for v in batch
            ]
            optimizer.zero_grad()
            out = model(
                graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                input_ids, attention_mask,
            )
            loss = fusion_multitask_loss(out, y_tags, y_val, y_aro, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate_fusion(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_micro_f1={metrics['micro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), "checkpoints/best_task3_concat.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (best macro_f1={best_f1:.4f})")
                break

    with open("results/metrics/task3_concat_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def _best_from_history(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        hist = json.load(f)
    if not hist:
        return None
    best = max(hist, key=lambda h: h.get("macro_f1", -1))
    return {
        "best_epoch": best.get("epoch"),
        "val_macro_f1": best.get("macro_f1"),
        "val_micro_f1": best.get("micro_f1"),
        "val_macro_auc_pr": best.get("macro_auc_pr"),
    }


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    print("=== Training concat-fusion ablation variant ===")
    train_concat_fusion(cfg)

    summary = {
        "bert_only (Task 1)": _best_from_history("results/metrics/task1_history.json"),
        "gnn_only (Task 2)": _best_from_history("results/metrics/task2_history.json"),
        "early_concat_fusion": _best_from_history("results/metrics/task3_concat_history.json"),
        "cross_attention_fusion (Task 3)": _best_from_history("results/metrics/task3_history.json"),
    }
    with open("results/metrics/ablations.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Ablation summary (best validation metrics) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
