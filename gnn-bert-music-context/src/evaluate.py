"""
evaluate.py
===========
All evaluation metrics required by the assignment rubric:
  - Macro-F1 / Micro-F1 (tag / genre classification)
  - AUC-PR (per tag, averaged)
  - MAE / R^2 (emotion regression, DEAM)
  - Graph coherence score (optional analysis, Section 6)
  - Retrieval R@K (Task 4, imported from contrastive.py)
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score, r2_score, mean_absolute_error


def compute_multilabel_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """
    y_true:       (N, K) binary ground-truth matrix
    y_pred_probs: (N, K) predicted probabilities (post-sigmoid)

    Returns dict with macro_f1, micro_f1, macro_auc_pr.
    """
    y_pred_bin = (y_pred_probs >= threshold).astype(int)

    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)

    # AUC-PR per tag, skipping tags with no positive examples in this split
    aucs = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() > 0:
            aucs.append(average_precision_score(y_true[:, k], y_pred_probs[:, k]))
    macro_auc_pr = float(np.mean(aucs)) if aucs else float("nan")

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "macro_auc_pr": macro_auc_pr}


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Standard regression metrics for valence or arousal.
    y_true, y_pred: (N,) arrays.
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "r2": r2}


def graph_coherence_score(node_embeddings: torch.Tensor, edge_index: torch.Tensor, tau: float = 0.85):
    """
    Optional analysis metric (Section 6 of the spec):
    fraction of edges (i, j) whose endpoint embeddings have cosine similarity > tau,
    i.e. whether high-attention edges correspond to genuinely similar / repeated content.

    node_embeddings: (num_nodes, dim)
    edge_index:      (2, num_edges)
    """
    src, dst = edge_index
    h_i = node_embeddings[src]
    h_j = node_embeddings[dst]

    cos_sim = torch.nn.functional.cosine_similarity(h_i, h_j, dim=-1)
    coherent = (cos_sim > tau).float()

    return coherent.mean().item()


def find_best_threshold(y_true: np.ndarray, y_pred_probs: np.ndarray, thresholds=None):
    """
    Sweeps thresholds to find the one maximizing macro-F1 on a validation set.
    Useful because 0.5 is rarely optimal for imbalanced multi-label tag data.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_thresh, best_f1 = 0.5, -1
    for t in thresholds:
        y_pred_bin = (y_pred_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh, best_f1


if __name__ == "__main__":
    # Smoke test with random data
    np.random.seed(0)
    y_true = (np.random.rand(200, 20) > 0.8).astype(int)
    y_pred_probs = np.random.rand(200, 20)

    metrics = compute_multilabel_metrics(y_true, y_pred_probs)
    print("Multilabel metrics (random data, sanity check only):", metrics)

    y_true_emotion = np.random.uniform(1, 9, 100)
    y_pred_emotion = y_true_emotion + np.random.normal(0, 0.5, 100)
    print("Emotion metrics (random data):", compute_emotion_metrics(y_true_emotion, y_pred_emotion))

    best_t, best_f1 = find_best_threshold(y_true, y_pred_probs)
    print(f"Best threshold: {best_t:.2f}, macro-F1: {best_f1:.4f}")
