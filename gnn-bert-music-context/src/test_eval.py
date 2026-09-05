"""
test_eval.py
============
Final held-out TEST-set evaluation of every trained model:

  - Task 1  BERT-only tag classifier          (checkpoints/best_task1.pt)
  - Task 2  GNN-only classifier               (checkpoints/best_task2.pt)
  - Task 3  GNN-BERT cross-attention fusion   (checkpoints/best_task3.pt)
  - Ablation: early-concat fusion             (checkpoints/best_task3_concat.pt)
  - Task 4  contrastive dual encoder          (checkpoints/best_task4.pt)
  - B1 random / majority baseline             (no checkpoint)
  - B2 CNN on mel-spectrograms                (checkpoints/best_cnn.pt)

Classification thresholds are tuned on the VALIDATION set (never on test).

Outputs:
  results/metrics.json                        <- headline table (spec Section 10)
  results/metrics/ablations.json              <- refreshed ablation summary
  results/metrics/embeddings_task3.npz        <- fused z + genres (for t-SNE plot)
  results/case_studies.json                   <- 3 test-track case studies
  results/retrieval_examples/retrieval_examples.json / .md  <- 10 qualitative examples

Usage (from repo root):
    python src/test_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

from baselines import MelSpecDataset
from bert_encoder import BertTagClassifier, BertTextEncoder
from contrastive import DualEncoder, retrieval_metrics
from datasets import MusicGraphCaptionDataset, MusicGraphDataset, MusicTagDataset
from evaluate import compute_multilabel_metrics, find_best_threshold
from fusion_model import GNNBertFusionModel
from gnn_model import CNNBaseline, GNNTagClassifier


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    want_cuda = cfg["device"] == "cuda"
    return torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")


def load_vocab(cfg):
    with open(Path(cfg["data"]["splits_dir"]) / "label_vocab.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_with_tuned_threshold(y_val, p_val, y_test, p_test):
    """Tune threshold on val, report test metrics at 0.5 and at the tuned value."""
    at_05 = compute_multilabel_metrics(y_test, p_test, threshold=0.5)
    best_t, _ = find_best_threshold(y_val, p_val)
    tuned = compute_multilabel_metrics(y_test, p_test, threshold=best_t)
    return {
        "threshold_0.5": at_05,
        "tuned_threshold": float(best_t),
        "at_tuned_threshold": tuned,
    }


# ---------------------------------------------------------------------------
# Probability collectors per model family
# ---------------------------------------------------------------------------
@torch.no_grad()
def probs_task1(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        P.append(torch.sigmoid(logits).cpu().numpy())
        Y.append(batch["labels"].numpy())
    return np.concatenate(Y), np.concatenate(P)


@torch.no_grad()
def probs_task2(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        P.append(torch.sigmoid(logits).cpu().numpy())
        Y.append(batch.y.cpu().numpy())
    return np.concatenate(Y), np.concatenate(P)


@torch.no_grad()
def probs_task3(model, loader, device, collect_z=False):
    model.eval()
    P, Y, Z = [], [], []
    for batch in loader:
        graph_batch, input_ids, attention_mask, y_tags = [
            v.to(device) if hasattr(v, "to") else v for v in batch[:4]
        ]
        out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                    input_ids, attention_mask)
        P.append(torch.sigmoid(out["tag_logits"]).cpu().numpy())
        Y.append(y_tags.cpu().numpy())
        if collect_z:
            Z.append(out["z"].cpu().numpy())
    z = np.concatenate(Z) if Z else None
    return np.concatenate(Y), np.concatenate(P), z


@torch.no_grad()
def probs_cnn(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        logits = model(batch["mel"].to(device))
        P.append(torch.sigmoid(logits).cpu().numpy())
        Y.append(batch["labels"].numpy())
    return np.concatenate(Y), np.concatenate(P)


# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    device = get_device(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    vocab = load_vocab(cfg)
    Path("results/retrieval_examples").mkdir(parents=True, exist_ok=True)

    tokenizer = BertTextEncoder(
        model_name=cfg["text"]["model_name"], finetune=False,
        max_length=cfg["text"]["max_length"],
    ).tokenizer
    bs = cfg["train"]["batch_size"]
    results = {}

    def loader_text(split):
        ds = MusicTagDataset(split=split, cfg=cfg, tokenizer=tokenizer)
        return ds, DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=ds.collate_fn)

    def loader_graph(split):
        ds = MusicGraphDataset(split=split, cfg=cfg)
        return ds, GeoDataLoader(ds, batch_size=bs, shuffle=False)

    def loader_pair(split, emotion=True):
        ds = MusicGraphCaptionDataset(split=split, cfg=cfg, tokenizer=tokenizer,
                                      include_emotion=emotion)
        return ds, DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=ds.collate_fn)

    # ------------------------------------------------------------------ Task 1
    ckpt = Path("checkpoints/best_task1.pt")
    if ckpt.exists():
        print("== Task 1: BERT-only ==")
        ds_val, val_loader = loader_text("val")
        ds_test, test_loader = loader_text("test")
        model = BertTagClassifier(
            model_name=cfg["text"]["model_name"], num_labels=ds_val.num_labels,
            finetune=False, max_length=cfg["text"]["max_length"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        yv, pv = probs_task1(model, val_loader, device)
        yt, pt = probs_task1(model, test_loader, device)
        results["task1_bert_only"] = _metrics_with_tuned_threshold(yv, pv, yt, pt)
        del model

    # ------------------------------------------------------------------ Task 2
    ckpt = Path("checkpoints/best_task2.pt")
    if ckpt.exists():
        print("== Task 2: GNN-only ==")
        ds_val, val_loader = loader_graph("val")
        ds_test, test_loader = loader_graph("test")
        model = GNNTagClassifier(
            in_dim=ds_val.node_feature_dim, num_labels=ds_val.num_labels,
            hidden_dim=cfg["gnn"]["hidden_dim"], out_dim=cfg["gnn"]["out_dim"],
            num_layers=cfg["gnn"]["num_layers"], dropout=cfg["gnn"]["dropout"],
            architecture=cfg["gnn"]["architecture"], gat_heads=cfg["gnn"]["gat_heads"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        yv, pv = probs_task2(model, val_loader, device)
        yt, pt = probs_task2(model, test_loader, device)
        results["task2_gnn_only"] = _metrics_with_tuned_threshold(yv, pv, yt, pt)
        del model

    # ------------------------------------------------------------------ Task 3 (+ concat ablation)
    def eval_fusion(ckpt_path, fusion_type, collect=False):
        ds_val, val_loader = loader_pair("val")
        ds_test, test_loader = loader_pair("test")
        model = GNNBertFusionModel(
            graph_in_dim=ds_val.node_feature_dim, num_labels=ds_val.num_labels,
            bert_model_name=cfg["text"]["model_name"],
            gnn_hidden_dim=cfg["gnn"]["hidden_dim"], gnn_out_dim=cfg["gnn"]["out_dim"],
            gnn_layers=cfg["gnn"]["num_layers"], gnn_architecture=cfg["gnn"]["architecture"],
            gnn_dropout=cfg["gnn"]["dropout"], fusion_type=fusion_type,
            fusion_hidden_dim=cfg["fusion"]["hidden_dim"], fusion_heads=cfg["fusion"]["num_heads"],
            bert_finetune=False, predict_emotion=True,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        yv, pv, _ = probs_task3(model, val_loader, device)
        yt, pt, z = probs_task3(model, test_loader, device, collect_z=collect)
        out = _metrics_with_tuned_threshold(yv, pv, yt, pt)
        return out, (yt, pt, z, ds_test.records), model

    ckpt = Path("checkpoints/best_task3.pt")
    task3_extras = None
    if ckpt.exists():
        print("== Task 3: GNN-BERT cross-attention fusion ==")
        res, task3_extras, model3 = eval_fusion(ckpt, cfg["fusion"]["type"], collect=True)
        results["task3_fusion_cross_attention"] = res

        # t-SNE inputs: fused z + genre of each test track
        yt, pt, z, records = task3_extras
        genres = [vocab[int(np.argmax(r["tags"]))] for r in records]
        np.savez("results/metrics/embeddings_task3.npz", z=z, genres=np.array(genres))

        # ---------------- case studies: 3 test tracks
        case_ids = [0, len(records) // 2, len(records) - 1]
        cases = []
        for i in case_ids:
            rec = records[i]
            probs = pt[i]
            top3 = np.argsort(probs)[::-1][:3]
            cases.append({
                "track_id": rec["track_id"],
                "caption": rec["text"],
                "true_genre": vocab[int(np.argmax(rec["tags"]))],
                "top3_predictions": [
                    {"genre": vocab[int(k)], "prob": float(probs[k])} for k in top3
                ],
            })
        with open("results/case_studies.json", "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
        del model3

    ckpt = Path("checkpoints/best_task3_concat.pt")
    if ckpt.exists():
        print("== Ablation: early-concat fusion ==")
        res, _, mc = eval_fusion(ckpt, "concat")
        results["task3_fusion_concat"] = res
        del mc

    # ------------------------------------------------------------------ Task 4
    ckpt = Path("checkpoints/best_task4.pt")
    if ckpt.exists():
        print("== Task 4: contrastive retrieval ==")
        ds_test, test_loader = loader_pair("test", emotion=False)
        model = DualEncoder(
            graph_in_dim=ds_test.node_feature_dim,
            bert_model_name=cfg["text"]["model_name"],
            gnn_hidden_dim=cfg["gnn"]["hidden_dim"], gnn_out_dim=cfg["gnn"]["out_dim"],
            gnn_layers=cfg["gnn"]["num_layers"],
            embed_dim=cfg["contrastive"]["embed_dim"], bert_finetune=False,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        G, T = [], []
        with torch.no_grad():
            for batch in test_loader:
                graph_batch, input_ids, attention_mask = [
                    v.to(device) if hasattr(v, "to") else v for v in batch[:3]
                ]
                g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                             input_ids, attention_mask)
                G.append(g)
                T.append(t)
        G, T = torch.cat(G), torch.cat(T)
        results["task4_contrastive_retrieval"] = retrieval_metrics(
            G, T, ks=tuple(cfg["eval"]["retrieval_k"])
        )

        # 10 qualitative caption -> top-3 audio examples
        sim = (T @ G.t()).cpu().numpy()      # caption i vs all graphs
        records = ds_test.records
        examples = []
        rng = np.random.RandomState(cfg["seed"])
        for qi in rng.choice(len(records), size=min(10, len(records)), replace=False):
            top3 = np.argsort(sim[qi])[::-1][:3]
            examples.append({
                "query_caption": records[qi]["text"],
                "query_track_id": records[qi]["track_id"],
                "query_genre": vocab[int(np.argmax(records[qi]["tags"]))],
                "top3_retrieved": [
                    {
                        "rank": r + 1,
                        "track_id": records[int(j)]["track_id"],
                        "genre": vocab[int(np.argmax(records[int(j)]["tags"]))],
                        "similarity": float(sim[qi, int(j)]),
                        "is_correct_pair": bool(int(j) == int(qi)),
                    }
                    for r, j in enumerate(top3)
                ],
            })
        with open("results/retrieval_examples/retrieval_examples.json", "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2)
        lines = ["# Task 4: 10 qualitative retrieval examples (caption -> top-3 audio)\n"]
        for e in examples:
            lines.append(f"## Query: {e['query_track_id']} ({e['query_genre']})")
            lines.append(f"> {e['query_caption']}\n")
            for m in e["top3_retrieved"]:
                mark = " <-- CORRECT PAIR" if m["is_correct_pair"] else ""
                lines.append(f"{m['rank']}. {m['track_id']} ({m['genre']}), sim={m['similarity']:.3f}{mark}")
            lines.append("")
        with open("results/retrieval_examples/retrieval_examples.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        del model

    # ------------------------------------------------------------------ Baselines
    print("== B1: random / majority ==")
    with open(Path(cfg["data"]["splits_dir"]) / "train.json", "r", encoding="utf-8") as f:
        y_train = np.array([r["tags"] for r in json.load(f)], dtype=np.float32)
    with open(Path(cfg["data"]["splits_dir"]) / "test.json", "r", encoding="utf-8") as f:
        test_recs = json.load(f)
    y_test = np.array([r["tags"] for r in test_recs], dtype=np.float32)
    marginals = y_train.mean(axis=0)
    p_test = np.tile(marginals, (len(y_test), 1))
    results["B1_random_marginal"] = compute_multilabel_metrics(y_test, p_test)

    ckpt = Path("checkpoints/best_cnn.pt")
    if ckpt.exists():
        print("== B2: CNN on mel-spectrograms ==")
        val_ds = MelSpecDataset("val", cfg)
        test_ds = MelSpecDataset("test", cfg)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        model = CNNBaseline(num_labels=val_ds.num_labels, n_mels=cfg["data"]["n_mels"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        yv, pv = probs_cnn(model, val_loader, device)
        yt, pt = probs_cnn(model, test_loader, device)
        results["B2_cnn_melspec"] = _metrics_with_tuned_threshold(yv, pv, yt, pt)
        del model

    # ------------------------------------------------------------------ save
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/metrics.json")
    print(json.dumps(results, indent=2))

    # refresh ablation summary from (now final) histories
    def _best(path):
        p = Path(path)
        if not p.exists():
            return None
        hist = json.load(open(p, encoding="utf-8"))
        if not hist:
            return None
        b = max(hist, key=lambda h: h.get("macro_f1", -1))
        return {"best_epoch": b.get("epoch"), "val_macro_f1": b.get("macro_f1"),
                "val_micro_f1": b.get("micro_f1"), "val_macro_auc_pr": b.get("macro_auc_pr")}

    summary = {
        "bert_only (Task 1)": _best("results/metrics/task1_history.json"),
        "gnn_only (Task 2)": _best("results/metrics/task2_history.json"),
        "early_concat_fusion": _best("results/metrics/task3_concat_history.json"),
        "cross_attention_fusion (Task 3)": _best("results/metrics/task3_history.json"),
    }
    with open("results/metrics/ablations.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Refreshed results/metrics/ablations.json")


if __name__ == "__main__":
    main()
