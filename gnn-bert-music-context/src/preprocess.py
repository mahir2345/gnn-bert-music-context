"""
preprocess.py
=============
Batch preprocessing for GTZAN-style audio under data/raw/:

  data/raw/<genre>/<genre>.XXXXX.wav   (or flat layout data/raw/<genre>.XXXXX.wav)

For each track:
  1. Build a segment-similarity graph -> data/processed/graphs/<track_id>.pt
     (3 s windows, 1.5 s hop, standardized-cosine similarity edges capped
      at 3 neighbors per node -- see graph_builder.py docstring)
  2. Save a pooled float16 log-mel spectrogram -> data/processed/mel/<track_id>.npy
     (CNN baseline input)
  3. Generate an AUDIO-DERIVED natural-language caption (BERT input).
  4. Write stratified train/val/test split JSONs under data/splits/.

IMPORTANT -- caption design (no label leakage):
    Captions are computed ONLY from the audio signal (tempo, key/mode
    estimate, spectral brightness, onset density, chord histogram, dynamics,
    spectral flatness). The genre label NEVER appears in the caption, so the
    text modality is an independent noisy view of the audio, not a copy of
    the target. This replaces an earlier genre-template caption scheme that
    leaked the label into the text input.

Split note: GTZAN ships no artist metadata, so artist-level splitting is not
possible; we use a stratified-by-genre random split (seeded) and state this
limitation in the report.

Usage (from repo root):
    python src/preprocess.py                 # full run
    python src/preprocess.py --limit 20      # quick sanity subset
    python src/preprocess.py --workers 7     # parallel feature extraction
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Allow `python src/preprocess.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

import librosa  # noqa: E402

from audio_features import extract_log_mel_pooled, load_audio  # noqa: E402
from graph_builder import (  # noqa: E402
    build_segment_similarity_graph,
    estimate_chord_sequence,
    save_graph,
)

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# Audio-derived caption generation (label-free by construction)
# ---------------------------------------------------------------------------
def _estimate_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Correlate mean chroma with rotated Krumhansl major/minor profiles."""
    best = (-np.inf, "C", "major")
    for root in range(12):
        rolled = np.roll(chroma_mean, -root)
        for mode, profile in [("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)]:
            c = np.corrcoef(rolled, profile)[0, 1]
            if c > best[0]:
                best = (c, _PITCH_CLASSES[root], mode)
    return best[1], best[2]


def _chord_name(label: str) -> str:
    """'A#min' -> 'A# minor', 'Cmaj' -> 'C major'."""
    if label.endswith("maj"):
        return f"{label[:-3]} major"
    if label.endswith("min"):
        return f"{label[:-3]} minor"
    return label


def generate_caption(y: np.ndarray, sr: int) -> str:
    """Two-sentence natural-language description computed only from audio."""
    # --- tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
    if tempo < 76:
        tempo_adj = "slow"
    elif tempo < 100:
        tempo_adj = "relaxed mid-tempo"
    elif tempo < 126:
        tempo_adj = "moderately paced"
    else:
        tempo_adj = "fast, energetic"

    # --- key / mode
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_root, key_mode = _estimate_key(chroma.mean(axis=1))

    # --- brightness (spectral centroid)
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    if centroid < 1200:
        bright = "a dark, bass-heavy timbre"
    elif centroid < 2000:
        bright = "a warm, rounded timbre"
    elif centroid < 3000:
        bright = "a bright timbre"
    else:
        bright = "a very bright, treble-heavy timbre"

    # --- onset density
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    rate = len(onsets) / (len(y) / sr)
    if rate < 1.0:
        onset_phrase = "sparse, widely spaced note onsets"
    elif rate < 2.5:
        onset_phrase = "a steady flow of note onsets"
    else:
        onset_phrase = "a dense stream of note onsets"

    # --- chord histogram (top 3)
    chords = estimate_chord_sequence(y, sr=sr)
    uniq, counts = np.unique(chords, return_counts=True)
    top = [_chord_name(c) for c in uniq[np.argsort(counts)[::-1]][:3]]
    while len(top) < 3:
        top.append(top[-1] if top else "C major")

    # --- texture: sustained vs percussive (flatness + onset rate)
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    if rate >= 2.0 and flatness < 0.05:
        texture = "balances sustained harmony with percussive attacks"
    elif rate >= 2.0:
        texture = "is dominated by percussive attacks"
    else:
        texture = "leans on sustained harmonic tones"

    # --- dynamics (RMS variation)
    rms = librosa.feature.rms(y=y)[0]
    dyn = float(rms.std() / (rms.mean() + 1e-8))
    if dyn < 0.25:
        dynamics = "loudness stays nearly constant across the clip"
    elif dyn < 0.5:
        dynamics = "loudness varies moderately across the clip"
    else:
        dynamics = "loudness swings strongly across the clip"

    # --- noisiness
    if flatness < 0.02:
        noise = "a clean, tonal spectrum"
    elif flatness < 0.08:
        noise = "mild noise in the spectrum"
    else:
        noise = "noticeable noise and distortion in the spectrum"

    article = "A" if tempo_adj[0].lower() not in "aeiou" else "An"
    s1 = (
        f"{article} {tempo_adj} recording at about {tempo:.0f} BPM in "
        f"{key_root} {key_mode}, with {bright} and {onset_phrase}."
    )
    s2 = (
        f"Its harmony centres on {top[0]}, {top[1]} and {top[2]}, "
        f"the texture {texture}, and {dynamics}, with {noise}."
    )
    return f"{s1} {s2}"


# ---------------------------------------------------------------------------
# Dataset discovery / splitting
# ---------------------------------------------------------------------------
def discover_audio_files(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """Returns list of (track_id, genre, path); genre from filename or folder."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".au"}
    found: list[tuple[str, str, Path]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts or path.name.startswith("."):
            continue
        stem = path.stem
        genre_from_name = stem.split(".")[0].lower()
        genre_from_parent = path.parent.name.lower()
        if genre_from_name in GTZAN_GENRES:
            genre = genre_from_name
        elif genre_from_parent in GTZAN_GENRES:
            genre = genre_from_parent
        else:
            genre = genre_from_name or genre_from_parent or "unknown"
        found.append((stem, genre, path))
    return found


def one_hot(genre: str, vocab: list[str]) -> list[int]:
    vec = [0] * len(vocab)
    if genre in vocab:
        vec[vocab.index(genre)] = 1
    return vec


def make_splits(records: list[dict], seed: int, ratios=(0.8, 0.1, 0.1)):
    """Stratified split by genre. GTZAN has no artist metadata, so artist-level
    splitting is impossible; this is documented as a limitation in the report."""
    rng = random.Random(seed)
    by_genre: dict[str, list[dict]] = {}
    for rec in records:
        by_genre.setdefault(rec["genre"], []).append(rec)

    train, val, test = [], [], []
    for group in by_genre.values():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(round(n * ratios[2]))) if n > 2 else 0
        n_val = max(1, int(round(n * ratios[1]))) if n > 1 else 0
        n_train = n - n_val - n_test
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Per-track worker (module-level so it pickles on Windows spawn)
# ---------------------------------------------------------------------------
_WORKER_CFG: dict = {}


def _init_worker(cfg: dict):
    global _WORKER_CFG
    _WORKER_CFG = cfg


def process_one(item: tuple[str, str, str]) -> dict | None:
    """(track_id, genre, path_str) -> record dict, or None on skip/failure."""
    track_id, genre, path_str = item
    cfg = _WORKER_CFG
    sr = cfg["data"]["sample_rate"]
    min_seconds = 10.0
    graphs_dir = Path(cfg["data"]["processed_dir"]) / "graphs"
    mel_dir = Path(cfg["data"]["processed_dir"]) / "mel"

    try:
        y = load_audio(path_str, sample_rate=sr)
        if len(y) < sr * min_seconds:
            return None

        graph = build_segment_similarity_graph(
            y, sr=sr,
            segment_seconds=cfg["data"]["segment_seconds"],
            hop_seconds=cfg["data"]["segment_hop_seconds"],
            similarity_threshold=cfg["data"]["similarity_threshold"],
            max_similarity_neighbors=cfg["data"].get("max_similarity_neighbors", 3),
        )
        save_graph(graph, str(graphs_dir / f"{track_id}.pt"))

        mel = extract_log_mel_pooled(
            y, sr=sr, n_mels=cfg["data"]["n_mels"],
            time_pool=cfg["data"].get("mel_time_pool", 4),
        )
        np.save(str(mel_dir / f"{track_id}.npy"), mel)

        caption = generate_caption(y, sr)
        return {
            "track_id": track_id,
            "genre": genre,
            "text": caption,
            "num_nodes": int(graph.x.shape[0]),
            "num_edges": int(graph.edge_index.shape[1]),
            "node_feature_dim": int(graph.x.shape[1]),
            "mel_shape": list(mel.shape),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {track_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess audio -> graphs + captions + splits")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N tracks (0 = all)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    (processed_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (processed_dir / "mel").mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    files = discover_audio_files(raw_dir)
    if not files:
        raise SystemExit(f"No audio files found under {raw_dir}.")

    observed = sorted({g for _, g, _ in files})
    vocab = list(GTZAN_GENRES) + [g for g in observed if g not in GTZAN_GENRES]

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"Found {len(files)} tracks across genres: {observed}")
    print(f"Label vocab ({len(vocab)}): {vocab}")
    print(f"Workers: {args.workers}")

    items = [(tid, g, str(p)) for tid, g, p in files]
    outputs = []
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for out in tqdm(pool.imap_unordered(process_one, items), total=len(items),
                            desc="Preprocessing"):
                outputs.append(out)
    else:
        _init_worker(cfg)
        for item in tqdm(items, desc="Preprocessing"):
            outputs.append(process_one(item))

    results = [o for o in outputs if o is not None]
    if not results:
        raise SystemExit("No tracks were successfully preprocessed.")
    results.sort(key=lambda r: r["track_id"])

    records = [
        {
            "track_id": r["track_id"],
            "text": r["text"],
            "tags": one_hot(r["genre"], vocab),
            "genre": r["genre"],
            "valence": None,   # kept null: GTZAN has no emotion labels (fusion loss masks NaN)
            "arousal": None,
        }
        for r in results
    ]

    train, val, test = make_splits(records, seed=seed)

    def dump(name: str, rows: list[dict]):
        path = splits_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {path} ({len(rows)} tracks)")

    dump("train", train)
    dump("val", val)
    dump("test", test)

    nodes = np.array([r["num_nodes"] for r in results])
    edges = np.array([r["num_edges"] for r in results])
    density = edges / np.maximum(nodes * (nodes - 1), 1)
    meta = {
        "audio_dataset": "gtzan",
        "text_source": "audio_derived_captions",
        "num_labels": len(vocab),
        "label_vocab": vocab,
        "genres_observed": observed,
        "num_tracks": len(results),
        "num_skipped": len(items) - len(results),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "node_feature_dim": results[0]["node_feature_dim"],
        "segment_seconds": cfg["data"]["segment_seconds"],
        "segment_hop_seconds": cfg["data"]["segment_hop_seconds"],
        "similarity_threshold": cfg["data"]["similarity_threshold"],
        "max_similarity_neighbors": cfg["data"].get("max_similarity_neighbors", 3),
        "graph_stats": {
            "nodes_mean": float(nodes.mean()),
            "nodes_min": int(nodes.min()),
            "nodes_max": int(nodes.max()),
            "edges_mean": float(edges.mean()),
            "edges_min": int(edges.min()),
            "edges_max": int(edges.max()),
            "density_mean": float(density.mean()),
            "fraction_complete_graphs": float((density >= 0.999).mean()),
        },
        "mel_shape_example": results[0]["mel_shape"],
        "mel_time_pool": cfg["data"].get("mel_time_pool", 4),
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(splits_dir / "label_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print("Done.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
