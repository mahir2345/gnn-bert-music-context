"""
graph_builder.py
=================
Builds two kinds of music-structure graphs used across Tasks 2-4:

1. Segment-similarity graph:
   nodes = fixed-length time segments
   edges = temporal adjacency + cosine-similarity(MFCC/chroma) > tau

2. Chord-transition graph:
   nodes = unique chords (estimated from chroma via template matching)
   edges = observed transitions, weighted by transition count

Output format: torch_geometric.data.Data objects, saved as .pt files.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from audio_features import segment_track, segment_embedding, extract_chroma

# --- Minimal major/minor chord templates (12 pitch classes x 24 chords) ----
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_chord_templates():
    """Binary templates for 12 major + 12 minor triads over the chroma vector."""
    templates = {}
    major_intervals = [0, 4, 7]
    minor_intervals = [0, 3, 7]

    for root in range(12):
        maj = np.zeros(12)
        for i in major_intervals:
            maj[(root + i) % 12] = 1
        templates[f"{_PITCH_CLASSES[root]}maj"] = maj

        minr = np.zeros(12)
        for i in minor_intervals:
            minr[(root + i) % 12] = 1
        templates[f"{_PITCH_CLASSES[root]}min"] = minr

    return templates


_CHORD_TEMPLATES = _build_chord_templates()


def estimate_chord_sequence(y: np.ndarray, sr: int = 22050, hop_seconds: float = 1.0):
    """
    Naive template-matching chord estimation over 1-second chroma frames.
    Not a substitute for a proper chord-recognition model, but sufficient
    for building a chord-transition graph for this assignment.

    Returns
    -------
    List[str]  -- sequence of chord labels, one per hop window
    """
    chroma = extract_chroma(y, sr=sr)  # (12, n_frames)
    frames_per_hop = max(int(hop_seconds * sr / 512), 1)  # librosa default hop_length=512

    chords = []
    for start in range(0, chroma.shape[1], frames_per_hop):
        window = chroma[:, start:start + frames_per_hop]
        if window.shape[1] == 0:
            continue
        avg_vec = window.mean(axis=1)
        best_chord, best_score = None, -np.inf
        for name, template in _CHORD_TEMPLATES.items():
            score = np.dot(avg_vec, template) / (np.linalg.norm(avg_vec) * np.linalg.norm(template) + 1e-8)
            if score > best_score:
                best_score, best_chord = score, name
        chords.append(best_chord)

    return chords


def build_chord_transition_graph(chord_sequence, node_feature_dim: int = 24) -> Data:
    """
    nodes = unique chords appearing in the track
    edges = observed i -> j transitions, weighted by count
    node features = one-hot (or learned embedding lookup index) over the 24 chord vocabulary
    """
    unique_chords = sorted(set(chord_sequence))
    chord_to_idx = {c: i for i, c in enumerate(unique_chords)}

    edge_counts = {}
    for i in range(len(chord_sequence) - 1):
        a, b = chord_to_idx[chord_sequence[i]], chord_to_idx[chord_sequence[i + 1]]
        edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

    if edge_counts:
        edge_index = torch.tensor(list(edge_counts.keys()), dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(list(edge_counts.values()), dtype=torch.float)
    else:
        # degenerate case: <2 chords in the sequence -> no transitions
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)

    # one-hot features over the full 24-chord vocabulary (order-independent, fixed dim)
    full_vocab = sorted(_CHORD_TEMPLATES.keys())
    vocab_to_idx = {c: i for i, c in enumerate(full_vocab)}
    x = torch.zeros((len(unique_chords), node_feature_dim))
    for chord, local_idx in chord_to_idx.items():
        x[local_idx, vocab_to_idx[chord]] = 1.0

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


def build_segment_similarity_graph(
    y: np.ndarray,
    sr: int = 22050,
    segment_seconds: float = 3.0,
    hop_seconds: float = 1.5,
    similarity_threshold: float = 0.5,
    max_similarity_neighbors: int = 3,
) -> Data:
    """
    nodes = fixed-length (overlapping) segments
    edges = temporal adjacency (i, i+1) UNION top-k cosine-similarity edges
    node features = segment_embedding() (mean-pooled mel + chroma)

    Design notes (these matter -- see report Section "Graph construction"):
      * Overlapping 3s/1.5s windows give ~19 nodes per 30s GTZAN clip instead
        of 6 non-overlapping 5s windows, so the graph has usable structure.
      * Cosine similarity is computed on PER-DIMENSION STANDARDIZED features;
        raw mean-pooled log-mel vectors are so correlated that nearly every
        pair exceeds any threshold, which degenerates into a complete graph
        (message passing == mean pooling). Standardizing first fixes that.
      * Each node keeps at most `max_similarity_neighbors` similarity edges,
        bounding graph density (~0.2 instead of ~0.95).
    """
    segments = segment_track(y, sr=sr, segment_seconds=segment_seconds, hop_seconds=hop_seconds)
    if len(segments) < 2:
        # pad with a duplicate so we always have a valid (if trivial) graph
        segments = segments * 2

    node_feats = np.stack([segment_embedding(s, sr=sr) for s in segments])  # (N, D)
    x = torch.tensor(node_feats, dtype=torch.float)

    n = len(segments)
    edges = set()

    # temporal adjacency
    for i in range(n - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))

    # similarity edges on standardized features, capped at top-k per node
    mu = node_feats.mean(axis=0, keepdims=True)
    sd = node_feats.std(axis=0, keepdims=True) + 1e-8
    zfeats = (node_feats - mu) / sd
    norm_feats = zfeats / (np.linalg.norm(zfeats, axis=1, keepdims=True) + 1e-8)
    sim_matrix = norm_feats @ norm_feats.T
    np.fill_diagonal(sim_matrix, -np.inf)

    for i in range(n):
        # candidate neighbors sorted by similarity, best first
        order = np.argsort(sim_matrix[i])[::-1]
        added = 0
        for j in order:
            if added >= max_similarity_neighbors:
                break
            if sim_matrix[i, j] <= similarity_threshold:
                break
            if abs(i - int(j)) == 1:
                continue  # already a temporal edge
            edges.add((i, int(j)))
            edges.add((int(j), i))
            added += 1

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def save_graph(data: Data, path: str):
    torch.save(data, path)


def load_graph(path: str) -> Data:
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    sr = 22050
    t = np.linspace(0, 20, sr * 20)
    y_test = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)

    chords = estimate_chord_sequence(y_test, sr=sr)
    chord_graph = build_chord_transition_graph(chords)
    print(f"Chord graph: {chord_graph}")

    seg_graph = build_segment_similarity_graph(y_test, sr=sr)
    print(f"Segment graph: {seg_graph}")
