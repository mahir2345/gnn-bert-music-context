"""
baselines.py
============
The two required baselines (spec Section 8):

  B1: Random / majority-class tag predictor
      Predicts the training-set marginal frequency of every tag for
      every track (constant prediction). No learning.

  B2: CNN on log-mel spectrograms (no graph, no text)
      Trains gnn_model.CNNBaseline on the saved data/processed/mel/*.npy
      spectrograms with the same BCE recipe as Tasks 1-2.

Outputs:
  checkpoints/best_cnn.pt
  results/metrics/cnn_history.json
  results/metrics/baselines.json     (validation metrics for B1 + B2)

Usage (from repo root):
    python src/baselines.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

from evaluate import compute_multilabel_metrics, find_best_threshold
from gnn_model import CNNBaseline


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    want_cuda = cfg["device"] == "cuda"
    return torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")


def load_split(cfg, split):
    with open(Path(cfg["data"]["splits_dir"]) / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# B1: random / majority baseline
# ---------------------------------------------------------------------------
def random_baseline(cfg):
    train = load_split(cfg, "train")
    val = load_split(cfg, "val")

    y_train = np.array([r["tags"] for r in train], dtype=np.float32)
    y_val = np.array([r["tags"] for r in val], dtype=np.float32)

    marginals = y_train.mean(axis=0)                       # (K,)
    probs = np.tile(marginals, (len(y_val), 1))            # constant prediction

    metrics = compute_multilabel_metrics(y_val, probs)

    # majority-class variant: predict 1 only for the most frequent tag
    majority = np.zeros_like(marginals)
    majority[marginals.argmax()] = 1.0
    maj_probs = np.tile(majority, (len(y_val), 1))
    maj_metrics = compute_multilabel_metrics(y_val, maj_probs)

    return {
        "marginal_frequency_predictor": metrics,
        "majority_class_predictor": maj_metrics,
        "train_tag_marginals": marginals.tolist(),
    }


# ---------------------------------------------------------------------------
# B2: CNN on mel-spectrograms
# ---------------------------------------------------------------------------
class MelSpecDataset(Dataset):
    """Loads data/processed/mel/<track_id>.npy, pads/crops time axis to n_frames."""

    def __init__(self, split: str, cfg: dict, n_frames: int = 323):
        self.records = load_split(cfg, split)
        self.mel_dir = Path(cfg["data"]["processed_dir"]) / "mel"
        self.n_frames = n_frames
        self.num_labels = len(self.records[0]["tags"]) if self.records else 0
        # drop records whose mel file is missing
        self.records = [r for r in self.records if (self.mel_dir / f"{r['track_id']}.npy").exists()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        mel = np.load(self.mel_dir / f"{rec['track_id']}.npy").astype(np.float32)
        t = mel.shape[1]
        if t < self.n_frames:
            mel = np.pad(mel, ((0, 0), (0, self.n_frames - t)))
        elif t > self.n_frames:
            mel = mel[:, : self.n_frames]
        return {
            "mel": torch.from_numpy(mel).unsqueeze(0),     # (1, n_mels, n_frames)
            "labels": torch.tensor(rec["tags"], dtype=torch.float),
        }


@torch.no_grad()
def eval_cnn(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logits = model(batch["mel"].to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    if not all_probs:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "macro_auc_pr": float("nan")}
    y, p = np.concatenate(all_labels), np.concatenate(all_probs)
    best_t, _ = find_best_threshold(y, p)
    metrics = compute_multilabel_metrics(y, p, threshold=best_t)
    metrics["threshold"] = float(best_t)
    return metrics


def train_cnn(cfg):
    device = get_device(cfg)
    train_ds = MelSpecDataset("train", cfg)
    val_ds = MelSpecDataset("val", cfg)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    print(f"CNN baseline: {len(train_ds)} train / {len(val_ds)} val tracks")

    model = CNNBaseline(num_labels=train_ds.num_labels, n_mels=cfg["data"]["n_mels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    patience = cfg["train"].get("early_stopping_patience", 5)
    best_f1, bad_epochs = -1.0, 0
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[CNN] Epoch {epoch+1}"):
            optimizer.zero_grad()
            logits = model(batch["mel"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = eval_cnn(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_micro_f1={metrics['micro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad_epochs = 0
            torch.save(model.state_dict(), "checkpoints/best_cnn.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (best macro_f1={best_f1:.4f})")
                break

    with open("results/metrics/cnn_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best = max(history, key=lambda h: h["macro_f1"])
    return {
        "best_epoch": best["epoch"],
        "val_macro_f1": best["macro_f1"],
        "val_micro_f1": best["micro_f1"],
        "val_macro_auc_pr": best["macro_auc_pr"],
    }


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    print("=== B1: random / majority baseline ===")
    b1 = random_baseline(cfg)
    print(json.dumps({k: v for k, v in b1.items() if k != "train_tag_marginals"}, indent=2))

    print("=== B2: CNN on mel-spectrograms ===")
    b2 = train_cnn(cfg)
    print(json.dumps(b2, indent=2))

    out = {"B1_random_majority": b1, "B2_cnn_melspec": b2}
    with open("results/metrics/baselines.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved results/metrics/baselines.json")


if __name__ == "__main__":
    main()
