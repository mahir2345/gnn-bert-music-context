"""
End-to-end smoke test: synthetic audio -> graphs -> Dataset classes ->
one training epoch for all 4 tasks. No real dataset needed.

Run from repo root:  python tests/smoke_test.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_builder import build_segment_similarity_graph, save_graph
from datasets import MusicTagDataset, MusicGraphDataset, MusicGraphCaptionDataset
from train import load_config, train_task1, train_task2, train_task3, train_task4
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader


def make_synthetic_data(tmp_dir, n_tracks=4, num_tags=8):
    """Generate sine-wave tracks, build their graphs, and write split JSONs."""
    graphs_dir = os.path.join(tmp_dir, "processed", "graphs")
    splits_dir = os.path.join(tmp_dir, "splits")
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    sr = 22050
    rng = np.random.default_rng(0)
    records = []
    for i in range(n_tracks):
        freq = 220 * (i + 1)
        t = np.linspace(0, 12, sr * 12)
        y = 0.5 * np.sin(2 * np.pi * freq * t)
        graph = build_segment_similarity_graph(y, sr=sr)
        save_graph(graph, os.path.join(graphs_dir, f"track{i}.pt"))

        tags = rng.integers(0, 2, num_tags).tolist()
        records.append({
            "track_id": f"track{i}",
            "text": f"synthetic tone at {freq} hertz, test caption {i}",
            "tags": tags,
            # half the tracks have no emotion labels -> tests NaN masking
            **({"valence": 5.0 + i, "arousal": 4.0 + i} if i % 2 == 0 else {}),
        })

    for split in ("train", "val"):
        with open(os.path.join(splits_dir, f"{split}.json"), "w") as f:
            json.dump(records, f)

    return graphs_dir, splits_dir


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Building synthetic dataset...")
        make_synthetic_data(tmp_dir)
        cfg["data"]["processed_dir"] = os.path.join(tmp_dir, "processed")
        cfg["data"]["splits_dir"] = os.path.join(tmp_dir, "splits")

        # ---- Task 1: BERT tag classifier ------------------------------
        print("\n[Task 1] BERT tag classifier")
        ds1 = MusicTagDataset("train", cfg)
        from bert_encoder import BertTextEncoder
        ds1.tokenizer = BertTextEncoder(cfg["text"]["model_name"]).tokenizer
        loader1 = DataLoader(ds1, batch_size=2, shuffle=True, collate_fn=ds1.collate_fn)
        train_task1(cfg, loader1, loader1, num_labels=ds1.num_labels)

        # ---- Task 2: GNN classifier ------------------------------------
        print("\n[Task 2] GNN tag classifier")
        ds2 = MusicGraphDataset("train", cfg)
        loader2 = GeoDataLoader(ds2, batch_size=2, shuffle=True)
        train_task2(cfg, loader2, loader2, num_labels=ds2.num_labels,
                    graph_in_dim=ds2.node_feature_dim)

        # ---- Task 3: fusion (with NaN emotion labels) ------------------
        print("\n[Task 3] GNN-BERT fusion")
        ds3 = MusicGraphCaptionDataset("train", cfg, include_emotion=True)
        ds3.tokenizer = ds1.tokenizer
        loader3 = DataLoader(ds3, batch_size=2, shuffle=True, collate_fn=ds3.collate_fn)
        model3 = train_task3(cfg, loader3, loader3, num_labels=ds3.num_labels,
                             graph_in_dim=ds3.node_feature_dim)
        for p in model3.parameters():
            assert not torch.isnan(p).any(), "NaN in Task 3 weights (emotion masking failed)"

        # ---- Task 4: contrastive ----------------------------------------
        print("\n[Task 4] Contrastive dual-encoder")
        ds4 = MusicGraphCaptionDataset("train", cfg, include_emotion=False)
        ds4.tokenizer = ds1.tokenizer
        loader4 = DataLoader(ds4, batch_size=2, shuffle=True, collate_fn=ds4.collate_fn)
        cfg["eval"]["retrieval_k"] = (1, 2)
        train_task4(cfg, loader4, loader4, graph_in_dim=ds4.node_feature_dim)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
