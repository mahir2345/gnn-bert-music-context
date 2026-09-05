"""
make_plots.py
=============
Generates every figure required by the spec (Section 6 / deliverables):

  results/plots/f1_curves.png            Macro/Micro-F1 vs epoch (Tasks 1-3, concat, CNN)
  results/plots/loss_curves.png          Training loss vs epoch
  results/plots/model_comparison.png     Test Macro-F1 / AUC-PR bar chart, all models
  results/plots/task4_retrieval.png      Val R@K vs epoch (Task 4)
  results/plots/tsne_fusion_z.png        t-SNE of fused embedding z, colored by genre
  results/plots/case_study_<i>_<id>.png  Segment graph + caption + predictions (3 test tracks)

Run AFTER src/test_eval.py.

Usage (from repo root):
    python src/make_plots.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

PLOTS = Path("results/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

HISTORIES = {
    "Task 1: BERT-only": "results/metrics/task1_history.json",
    "Task 2: GNN-only": "results/metrics/task2_history.json",
    "Task 3: cross-attn fusion": "results/metrics/task3_history.json",
    "Ablation: concat fusion": "results/metrics/task3_concat_history.json",
    "B2: CNN mel-spec": "results/metrics/cnn_history.json",
}


def _load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_f1_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, path in HISTORIES.items():
        hist = _load(path)
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["macro_f1"] for h in hist], marker="o", ms=3, label=label)
        axes[1].plot(epochs, [h["micro_f1"] for h in hist], marker="o", ms=3, label=label)
    axes[0].set_title("Validation Macro-F1 vs epoch")
    axes[1].set_title("Validation Micro-F1 vs epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "f1_curves.png", dpi=150)
    plt.close(fig)
    print("saved f1_curves.png")


def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in HISTORIES.items():
        hist = _load(path)
        if not hist:
            continue
        ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist],
                marker="o", ms=3, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (BCE / multitask)")
    ax.set_title("Training loss vs epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "loss_curves.png", dpi=150)
    plt.close(fig)
    print("saved loss_curves.png")


def plot_model_comparison():
    metrics = _load("results/metrics.json")
    if not metrics:
        print("results/metrics.json missing - run src/test_eval.py first")
        return
    rows = []
    for key, label in [
        ("B1_random_marginal", "B1 Random"),
        ("B2_cnn_melspec", "B2 CNN mel-spec"),
        ("task1_bert_only", "Task 1 BERT-only"),
        ("task2_gnn_only", "Task 2 GNN-only"),
        ("task3_fusion_concat", "Concat fusion"),
        ("task3_fusion_cross_attention", "Task 3 GNN-BERT"),
    ]:
        m = metrics.get(key)
        if m is None:
            continue
        inner = m.get("at_tuned_threshold", m)   # B1 has flat structure
        rows.append((label, inner["macro_f1"], inner["macro_auc_pr"]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [r[1] for r in rows], width=0.4, label="Macro-F1 (test)")
    ax.bar(x + 0.2, [r[2] for r in rows], width=0.4, label="AUC-PR (test)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Test-set comparison (thresholds tuned on validation)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r[1] + 0.01, f"{r[1]:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, r[2] + 0.01, f"{r[2]:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=150)
    plt.close(fig)
    print("saved model_comparison.png")


def plot_task4_retrieval():
    hist = _load("results/metrics/task4_history.json")
    if not hist:
        return
    ks = [k for k in hist[-1] if k.startswith("caption_to_audio_R@")]
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = [h["epoch"] for h in hist]
    for key in sorted(ks, key=lambda s: int(s.split("@")[1])):
        ax.plot(epochs, [h.get(key, np.nan) for h in hist], marker="o", ms=3,
                label=key.replace("caption_to_audio_", "Caption->Audio "))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall@K (validation)")
    ax.set_title("Task 4: contrastive retrieval, validation R@K vs epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "task4_retrieval.png", dpi=150)
    plt.close(fig)
    print("saved task4_retrieval.png")


def plot_tsne():
    p = Path("results/metrics/embeddings_task3.npz")
    if not p.exists():
        print("embeddings_task3.npz missing - run src/test_eval.py first")
        return
    from sklearn.manifold import TSNE

    data = np.load(p, allow_pickle=True)
    z, genres = data["z"], data["genres"]
    perplexity = max(5, min(30, len(z) // 4))
    z2 = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 7))
    for g in sorted(set(genres.tolist())):
        mask = genres == g
        ax.scatter(z2[mask, 0], z2[mask, 1], s=25, alpha=0.75, label=g)
    ax.set_title("t-SNE of fused GNN-BERT embedding z (test set), colored by genre")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "tsne_fusion_z.png", dpi=150)
    plt.close(fig)
    print("saved tsne_fusion_z.png")


def plot_case_studies():
    cases = _load("results/case_studies.json")
    if not cases:
        print("case_studies.json missing - run src/test_eval.py first")
        return
    from graph_builder import load_graph

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    graphs_dir = Path(cfg["data"]["processed_dir"]) / "graphs"

    for i, case in enumerate(cases):
        gpath = graphs_dir / f"{case['track_id']}.pt"
        if not gpath.exists():
            continue
        g = load_graph(str(gpath))
        G = nx.Graph()
        G.add_nodes_from(range(g.x.shape[0]))
        edges = g.edge_index.t().tolist()
        G.add_edges_from([(a, b) for a, b in edges if a < b])

        fig, ax = plt.subplots(figsize=(9, 7))
        pos = nx.spring_layout(G, seed=42)
        temporal = {(a, b) for a, b in G.edges() if abs(a - b) == 1}
        similarity = [e for e in G.edges() if e not in temporal]
        nx.draw_networkx_edges(G, pos, edgelist=list(temporal), ax=ax, edge_color="steelblue", width=2)
        nx.draw_networkx_edges(G, pos, edgelist=similarity, ax=ax, edge_color="orange",
                               style="dashed", alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightgray", edgecolors="black", node_size=380)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        preds = ", ".join(f"{p['genre']} ({p['prob']:.2f})" for p in case["top3_predictions"])
        caption = case["caption"]
        wrapped = "\n".join(caption[j:j + 95] for j in range(0, len(caption), 95))
        ax.set_title(
            f"Case study: {case['track_id']}  (true: {case['true_genre']})\n"
            f"Top-3 predictions: {preds}\n\nCaption: {wrapped}",
            fontsize=8, loc="left",
        )
        ax.text(0.01, -0.06,
                "blue solid = temporal adjacency edges   |   orange dashed = chroma/mel similarity edges",
                transform=ax.transAxes, fontsize=7)
        ax.axis("off")
        fig.tight_layout()
        fname = PLOTS / f"case_study_{i+1}_{case['track_id']}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname.name}")


def main():
    plot_f1_curves()
    plot_loss_curves()
    plot_model_comparison()
    plot_task4_retrieval()
    plot_tsne()
    plot_case_studies()
    print("All plots written to results/plots/")


if __name__ == "__main__":
    main()
