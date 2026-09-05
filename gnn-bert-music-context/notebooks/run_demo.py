"""
Run the demo without Jupyter (avoids kernel issues).

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe notebooks\\run_demo.py
    .\\.venv\\Scripts\\python.exe notebooks\\run_demo.py --audio data\\raw\\jazz.00001.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from torch_geometric.data import Batch

from audio_features import load_audio
from fusion_model import GNNBertFusionModel
from graph_builder import build_segment_similarity_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="data/raw/jazz.00000.wav")
    parser.add_argument(
        "--caption",
        type=str,
        default="a jazz music track with improvisation and swing rhythm",
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/fusion_model.pt")
    args = parser.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_path = (ROOT / args.audio).resolve() if not Path(args.audio).is_absolute() else Path(args.audio)
    ckpt_path = (ROOT / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt)

    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    y = load_audio(str(audio_path), sample_rate=cfg["data"]["sample_rate"])
    graph = build_segment_similarity_graph(
        y,
        sr=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        hop_seconds=cfg["data"]["segment_hop_seconds"],
        similarity_threshold=cfg["data"]["similarity_threshold"],
    )
    print("graph:", graph)

    vocab_path = ROOT / "data" / "splits" / "label_vocab.json"
    if vocab_path.exists():
        TAG_VOCAB = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        TAG_VOCAB = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock",
        ]

    model = GNNBertFusionModel(
        graph_in_dim=graph.x.shape[1],
        num_labels=len(TAG_VOCAB),
        bert_model_name=cfg["text"]["model_name"],
        gnn_hidden_dim=cfg["gnn"]["hidden_dim"],
        gnn_out_dim=cfg["gnn"]["out_dim"],
        gnn_layers=cfg["gnn"]["num_layers"],
        gnn_architecture=cfg["gnn"]["architecture"],
        fusion_type=cfg["fusion"]["type"],
        fusion_hidden_dim=cfg["fusion"]["hidden_dim"],
        bert_finetune=False,
        predict_emotion=True,
    ).to(device)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        print("Loaded:", ckpt_path)
    else:
        print("WARNING: checkpoint missing, random weights")

    model.eval()
    graph_batch = Batch.from_data_list([graph]).to(device)
    enc = model.bert.tokenize([args.caption])
    with torch.no_grad():
        out = model(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch,
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        probs = torch.sigmoid(out["tag_logits"]).cpu().numpy()[0]

    print("\nPredicted tags:")
    for tag, p in sorted(zip(TAG_VOCAB, probs), key=lambda x: -x[1]):
        print(f"  {tag:15s} {p:.3f}")
    if out["valence"] is not None:
        print(f"\nvalence={out['valence'].item():.2f}  arousal={out['arousal'].item():.2f}")


if __name__ == "__main__":
    main()
