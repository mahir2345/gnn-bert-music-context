"""
datasets.py
===========
PyTorch Dataset wrappers around the files produced by the preprocessing
pipeline (see README.md -> "Preprocessing" for the exact on-disk layout
expected under data/processed/ and data/splits/).

Expected layout
----------------
data/processed/
    graphs/<track_id>.pt          # torch_geometric.data.Data
    mel/<track_id>.npy            # optional, for CNN baseline
data/splits/
    train.json   # [{"track_id", "text", "tags", "valence", "arousal"}, ...]
    val.json
    test.json
    label_vocab.json
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from graph_builder import load_graph


def _as_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


class MusicTagDataset(Dataset):
    """Task 1: text-only dataset (tags/captions -> multi-label targets)."""

    def __init__(self, split: str, cfg: dict, tokenizer=None):
        split_path = os.path.join(cfg["data"]["splits_dir"], f"{split}.json")
        with open(split_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.num_labels = len(self.records[0]["tags"]) if self.records else 0
        self.tokenizer = tokenizer
        self.max_length = cfg.get("text", {}).get("max_length", 128)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "text": rec["text"],
            "labels": torch.tensor(rec["tags"], dtype=torch.float),
            "track_id": rec["track_id"],
        }

    def collate_fn(self, batch):
        texts = [b["text"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch])
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


class MusicGraphDataset(Dataset):
    """Task 2: graph-only dataset (structure graph -> multi-label targets)."""

    def __init__(self, split: str, cfg: dict):
        split_path = os.path.join(cfg["data"]["splits_dir"], f"{split}.json")
        with open(split_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.graphs_dir = os.path.join(cfg["data"]["processed_dir"], "graphs")
        self.num_labels = len(self.records[0]["tags"]) if self.records else 0

        first_graph = load_graph(os.path.join(self.graphs_dir, f"{self.records[0]['track_id']}.pt"))
        self.node_feature_dim = first_graph.x.shape[1]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        graph = load_graph(os.path.join(self.graphs_dir, f"{rec['track_id']}.pt"))
        graph.y = torch.tensor(rec["tags"], dtype=torch.float).unsqueeze(0)
        return graph


class MusicGraphCaptionDataset(Dataset):
    """Task 3 / Task 4: paired (graph, text, [emotion]) dataset."""

    def __init__(self, split: str, cfg: dict, tokenizer=None, include_emotion: bool = True):
        split_path = os.path.join(cfg["data"]["splits_dir"], f"{split}.json")
        with open(split_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.graphs_dir = os.path.join(cfg["data"]["processed_dir"], "graphs")
        self.tokenizer = tokenizer
        self.include_emotion = include_emotion
        self.max_length = cfg.get("text", {}).get("max_length", 128)
        self.num_labels = len(self.records[0]["tags"]) if self.records else 0

        first_graph = load_graph(os.path.join(self.graphs_dir, f"{self.records[0]['track_id']}.pt"))
        self.node_feature_dim = first_graph.x.shape[1]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        graph = load_graph(os.path.join(self.graphs_dir, f"{rec['track_id']}.pt"))
        item = {
            "graph": graph,
            "text": rec["text"],
            "tags": torch.tensor(rec["tags"], dtype=torch.float),
            "track_id": rec["track_id"],
        }
        if self.include_emotion:
            item["valence"] = torch.tensor(_as_float_or_nan(rec.get("valence")), dtype=torch.float)
            item["arousal"] = torch.tensor(_as_float_or_nan(rec.get("arousal")), dtype=torch.float)
        return item

    def collate_fn(self, batch):
        from torch_geometric.data import Batch

        graphs = Batch.from_data_list([b["graph"] for b in batch])
        texts = [b["text"] for b in batch]
        tags = torch.stack([b["tags"] for b in batch])

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        out = (graphs, enc["input_ids"], enc["attention_mask"], tags)
        if self.include_emotion:
            valence = torch.stack([b["valence"] for b in batch])
            arousal = torch.stack([b["arousal"] for b in batch])
            out = out + (valence, arousal)
        return out
