"""
train.py
========
Unified training entry point for all four tasks:

    python src/train.py --task 1
    python src/train.py --task 2
    python src/train.py --task 3
    python src/train.py --task 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

# Allow `python src/train.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

from bert_encoder import BertTagClassifier, BertTextEncoder
from contrastive import DualEncoder, info_nce_loss, retrieval_metrics
from datasets import MusicGraphCaptionDataset, MusicGraphDataset, MusicTagDataset
from evaluate import compute_multilabel_metrics, find_best_threshold


def _tuned_metrics(y_true, y_probs):
    """Validation metrics with the classification threshold tuned on the same
    validation set (0.5 is rarely optimal for imbalanced multi-label data).
    The tuned threshold is recorded so test_eval.py can reuse the recipe."""
    best_t, _ = find_best_threshold(y_true, y_probs)
    metrics = compute_multilabel_metrics(y_true, y_probs, threshold=best_t)
    metrics["threshold"] = float(best_t)
    return metrics
from fusion_model import GNNBertFusionModel, fusion_multitask_loss
from gnn_model import GNNTagClassifier


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    want_cuda = cfg["device"] == "cuda"
    return torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")


def ensure_dirs():
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def eval_task1(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    if not all_probs:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "macro_auc_pr": float("nan")}
    return _tuned_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


@torch.no_grad()
def eval_task2(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    if not all_probs:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "macro_auc_pr": float("nan")}
    return _tuned_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


@torch.no_grad()
def eval_task3(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        graph_batch, input_ids, attention_mask, y_tags = [
            v.to(device) if hasattr(v, "to") else v for v in batch[:4]
        ]
        out = model(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch,
            input_ids,
            attention_mask,
        )
        all_probs.append(torch.sigmoid(out["tag_logits"]).cpu().numpy())
        all_labels.append(y_tags.cpu().numpy())
    if not all_probs:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "macro_auc_pr": float("nan")}
    return _tuned_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


def train_task1(cfg, train_loader, val_loader, num_labels):
    device = get_device(cfg)
    model = BertTagClassifier(
        model_name=cfg["text"]["model_name"],
        num_labels=num_labels,
        finetune=cfg["train"]["bert_finetune"],
        max_length=cfg["text"]["max_length"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    patience = cfg["train"].get("early_stopping_patience", 5)
    best_f1, bad_epochs = -1.0, 0
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Task1] Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = eval_task1(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_micro_f1={metrics['micro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), "checkpoints/best_task1.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (best macro_f1={best_f1:.4f})")
                break

    _save_json("results/metrics/task1_history.json", history)
    return model


def train_task2(cfg, train_loader, val_loader, num_labels, graph_in_dim):
    device = get_device(cfg)
    model = GNNTagClassifier(
        in_dim=graph_in_dim,
        num_labels=num_labels,
        hidden_dim=cfg["gnn"]["hidden_dim"],
        out_dim=cfg["gnn"]["out_dim"],
        num_layers=cfg["gnn"]["num_layers"],
        dropout=cfg["gnn"]["dropout"],
        architecture=cfg["gnn"]["architecture"],
        gat_heads=cfg["gnn"]["gat_heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    patience = cfg["train"].get("early_stopping_patience", 5)
    best_f1, bad_epochs = -1.0, 0
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Task2] Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = eval_task2(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_micro_f1={metrics['micro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), "checkpoints/best_task2.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (best macro_f1={best_f1:.4f})")
                break

    _save_json("results/metrics/task2_history.json", history)
    return model


def train_task3(cfg, train_loader, val_loader, num_labels, graph_in_dim):
    device = get_device(cfg)
    model = GNNBertFusionModel(
        graph_in_dim=graph_in_dim,
        num_labels=num_labels,
        bert_model_name=cfg["text"]["model_name"],
        gnn_hidden_dim=cfg["gnn"]["hidden_dim"],
        gnn_out_dim=cfg["gnn"]["out_dim"],
        gnn_layers=cfg["gnn"]["num_layers"],
        gnn_architecture=cfg["gnn"]["architecture"],
        gnn_dropout=cfg["gnn"]["dropout"],
        fusion_type=cfg["fusion"]["type"],
        fusion_hidden_dim=cfg["fusion"]["hidden_dim"],
        fusion_heads=cfg["fusion"]["num_heads"],
        bert_finetune=cfg["train"]["bert_finetune"],
        predict_emotion=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
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
        for batch in tqdm(train_loader, desc=f"[Task3] Epoch {epoch+1}"):
            graph_batch, input_ids, attention_mask, y_tags, y_val, y_aro = [
                v.to(device) if hasattr(v, "to") else v for v in batch
            ]
            optimizer.zero_grad()
            out = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.batch,
                input_ids,
                attention_mask,
            )
            loss = fusion_multitask_loss(out, y_tags, y_val, y_aro, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = eval_task3(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_micro_f1={metrics['micro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), "checkpoints/best_task3.pt")
            torch.save(model.state_dict(), "checkpoints/fusion_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (best macro_f1={best_f1:.4f})")
                break

    _save_json("results/metrics/task3_history.json", history)
    return model


@torch.no_grad()
def eval_task4(model, loader, device, ks=(1, 5, 10)):
    """Encode the full validation set and compute retrieval R@K on it."""
    model.eval()
    all_g, all_t = [], []
    for batch in loader:
        graph_batch, input_ids, attention_mask = [
            v.to(device) if hasattr(v, "to") else v for v in batch[:3]
        ]
        g, t = model(
            graph_batch.x, graph_batch.edge_index, graph_batch.batch,
            input_ids, attention_mask,
        )
        all_g.append(g)
        all_t.append(t)
    if not all_g:
        return {}
    return retrieval_metrics(torch.cat(all_g), torch.cat(all_t), ks=ks)


def train_task4(cfg, train_loader, val_loader, graph_in_dim):
    device = get_device(cfg)
    model = DualEncoder(
        graph_in_dim=graph_in_dim,
        bert_model_name=cfg["text"]["model_name"],
        gnn_hidden_dim=cfg["gnn"]["hidden_dim"],
        gnn_out_dim=cfg["gnn"]["out_dim"],
        gnn_layers=cfg["gnn"]["num_layers"],
        embed_dim=cfg["contrastive"]["embed_dim"],
        bert_finetune=cfg["train"]["bert_finetune"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    temperature = cfg["contrastive"]["temperature"]
    history = []
    best_r5 = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Task4] Epoch {epoch+1}"):
            graph_batch, input_ids, attention_mask = [
                v.to(device) if hasattr(v, "to") else v for v in batch[:3]
            ]
            optimizer.zero_grad()
            g, t = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.batch,
                input_ids,
                attention_mask,
            )
            loss = info_nce_loss(g, t, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = eval_task4(model, val_loader, device, ks=tuple(cfg["eval"]["retrieval_k"]))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_retrieval={metrics}")

        r5 = metrics.get("caption_to_audio_R@5", 0.0)
        if r5 >= best_r5:
            best_r5 = r5
            torch.save(model.state_dict(), "checkpoints/best_task4.pt")

    _save_json("results/metrics/task4_history.json", history)
    return model


def _get_tokenizer(cfg):
    return BertTextEncoder(
        model_name=cfg["text"]["model_name"],
        finetune=False,
        max_length=cfg["text"]["max_length"],
    ).tokenizer


def build_loaders(args, cfg):
    bs = cfg["train"]["batch_size"]

    if args.task == 2:
        train_ds = MusicGraphDataset(split="train", cfg=cfg)
        val_ds = MusicGraphDataset(split="val", cfg=cfg)
        train_loader = GeoDataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = GeoDataLoader(val_ds, batch_size=bs, shuffle=False)
        return train_loader, val_loader, train_ds.num_labels, train_ds.node_feature_dim

    tokenizer = _get_tokenizer(cfg)

    if args.task == 1:
        train_ds = MusicTagDataset(split="train", cfg=cfg, tokenizer=tokenizer)
        val_ds = MusicTagDataset(split="val", cfg=cfg, tokenizer=tokenizer)
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, collate_fn=train_ds.collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, collate_fn=val_ds.collate_fn
        )
        return train_loader, val_loader, train_ds.num_labels, None

    # Tasks 3 and 4
    include_emotion = args.task == 3
    train_ds = MusicGraphCaptionDataset(
        split="train", cfg=cfg, tokenizer=tokenizer, include_emotion=include_emotion
    )
    val_ds = MusicGraphCaptionDataset(
        split="val", cfg=cfg, tokenizer=tokenizer, include_emotion=include_emotion
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=train_ds.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, collate_fn=val_ds.collate_fn
    )
    return train_loader, val_loader, train_ds.num_labels, train_ds.node_feature_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    ensure_dirs()

    split_train = Path(cfg["data"]["splits_dir"]) / "train.json"
    if not split_train.exists() or split_train.stat().st_size == 0:
        raise SystemExit(
            "Missing/empty data/splits/train.json. "
            "Run: python src/preprocess.py"
        )

    device = get_device(cfg)
    print(f"Running Task {args.task} on {device} with config: {args.config}")

    train_loader, val_loader, num_labels, graph_in_dim = build_loaders(args, cfg)
    print(f"num_labels={num_labels}, graph_in_dim={graph_in_dim}")
    print(f"train batches={len(train_loader)}, val batches={len(val_loader)}")

    if args.task == 1:
        train_task1(cfg, train_loader, val_loader, num_labels=num_labels)
    elif args.task == 2:
        train_task2(
            cfg, train_loader, val_loader, num_labels=num_labels, graph_in_dim=graph_in_dim
        )
    elif args.task == 3:
        train_task3(
            cfg, train_loader, val_loader, num_labels=num_labels, graph_in_dim=graph_in_dim
        )
    elif args.task == 4:
        train_task4(cfg, train_loader, val_loader, graph_in_dim=graph_in_dim)

    print("Training finished. Checkpoints in checkpoints/, metrics in results/metrics/.")


if __name__ == "__main__":
    main()

