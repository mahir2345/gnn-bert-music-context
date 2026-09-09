# GNN-BERT Music Context Understanding

CSE425 / EEE474 / CSE715 — Neural Networks (deadline: 2 October 2026)

A hybrid **BERT + Graph Neural Network** system for music context understanding on
**GTZAN** (999 tracks, 10 genres). All four tasks from the assignment spec are
implemented and trained end-to-end:

| Model | Entry point |
|---|---|
| BERT multi-label tag classifier | `python src/train.py --task 1` |
| GraphSAGE on segment-similarity graphs | `python src/train.py --task 2` |
| GNN-BERT cross-attention fusion (+ concat ablation) | `python src/train.py --task 3`, `python src/ablations.py` |
| Contrastive dual-encoder, caption↔audio retrieval | `python src/train.py --task 4` |

Baselines (spec Section 8): **B1** random/marginal predictor and **B2** CNN on
mel-spectrograms (`python src/baselines.py`), plus BERT-only (Task 1) and GNN-only
(Task 2) as ablation rows.

---

## 1. Data design (read this first)

**Audio source:** GTZAN under `data/raw/<genre>/<genre>.XXXXX.wav`
(1,000 clips of 30 s; one corrupt file is skipped → 999 tracks).

**Text source: audio-derived captions.** GTZAN has no captions or tags, so
`src/preprocess.py::generate_caption` writes a two-sentence natural-language
description for every track computed **only from the waveform**: tempo, estimated
key/mode (Krumhansl profiles), spectral brightness, onset density, top-3 chords
(template matching over chroma), texture, dynamics, and noisiness. Example:

> "A relaxed mid-tempo recording at about 83 BPM in D minor, with a very bright,
> treble-heavy timbre and a dense stream of note onsets. Its harmony centres on
> G minor, A# minor and D minor, …"

The genre label **never** appears in the caption, so the text branch is an
independent noisy view of the audio — *not* a lookup table of the target. (An
earlier version of this repo used one template sentence per genre; that leaks the
label into the input and produces a meaningless perfect score. `notebooks/eda.ipynb`
Section 4 verifies captions are ~unique per track.)

**Graphs:** nodes = overlapping 3 s segments (1.5 s hop, ~19 per clip) with
140-dim features (mean-pooled log-mel 128 + chroma 12). Edges = temporal adjacency
∪ top-3 cosine-similarity neighbors above τ=0.5 **on per-dimension standardized
features**. Standardization + the neighbor cap keep density ≈ 0.21; without them,
raw mel vectors are so correlated that graphs become complete and message passing
degenerates to mean pooling.

**Splits:** stratified by genre, 80/10/10, seed 42 (`data/splits/*.json`).
GTZAN ships no artist metadata, so artist-level splitting is impossible — this is
a known limitation of the dataset and is stated in the report.

**Emotion heads:** the Task 3 model has valence/arousal regression heads and a
NaN-masked multitask loss, but GTZAN has no emotion annotations, so all targets
are null and the emotion term never activates. Wiring in DEAM would light it up
without code changes.

---

## 2. Setup

```bash
cd gnn-bert-music-context
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

If `torch-geometric` fails to install, install PyTorch first from
https://pytorch.org/get-started/locally/ and then follow
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

---

## 3. Reproducing everything

```bash
# 1. Preprocess: graphs + mel spectrograms + captions + splits  (~20 min, 7 workers)
python src/preprocess.py --workers 7

# 2. Train (CPU-friendly; DistilBERT frozen, classification heads trained)
python src/train.py --task 1
python src/train.py --task 2
python src/train.py --task 4 --epochs 15
python src/ablations.py            # concat-fusion ablation variant
python src/train.py --task 3
python src/baselines.py            # B1 random + B2 CNN on mel-spectrograms

# 3. Final held-out test evaluation + figures
python src/test_eval.py            # -> results/metrics.json, retrieval examples, case studies
python src/make_plots.py           # -> results/plots/*.png
```

Model selection uses validation macro-F1 with the classification threshold tuned
on the validation set (`evaluate.find_best_threshold`); the test set is touched
exactly once, by `src/test_eval.py`.

### Module map

| File | Contents |
|---|---|
| `src/audio_features.py` | mel/chroma extraction, segmentation, pooled float16 mel export |
| `src/graph_builder.py` | segment-similarity graph (standardized cosine, top-k), chord-transition graph |
| `src/preprocess.py` | batch pipeline: graphs + mels + audio-derived captions + splits |
| `src/datasets.py` | `MusicTagDataset`, `MusicGraphDataset`, `MusicGraphCaptionDataset` |
| `src/bert_encoder.py` | `BertTextEncoder`, `BertTagClassifier` (Task 1) |
| `src/gnn_model.py` | GraphSAGE / GAT encoders, `GNNTagClassifier` (Task 2), `CNNBaseline` |
| `src/fusion_model.py` | cross-attention + concat fusion, NaN-masked multitask loss (Task 3) |
| `src/contrastive.py` | dual encoder, symmetric InfoNCE, Recall@K (Task 4) |
| `src/train.py` | training loops for all 4 tasks, early stopping, history logging |
| `src/ablations.py` | concat-fusion variant + ablation summary table |
| `src/baselines.py` | B1 random/marginal + B2 CNN on mel-spectrograms |
| `src/test_eval.py` | one-shot test-set evaluation of every model → `results/metrics.json` |
| `src/make_plots.py` | F1 curves, model comparison, t-SNE, retrieval curve, case studies |

---

## 4. Results artifacts

- `results/metrics.json` — headline test-set table (all models, tuned thresholds)
- `results/metrics/*.json` — per-epoch training histories, ablation + baseline summaries
- `results/plots/` — F1/loss curves, model comparison bars, t-SNE of fused `z`,
  Task 4 retrieval curve, 3 graph case studies
- `results/retrieval_examples/` — 10 qualitative caption→top-3-audio examples
- `results/case_studies.json` — case-study predictions backing the figures
- `data/processed/metadata.json` — dataset + graph construction statistics
- `report/final_report.pdf` — final report

## 5. Notebooks

- `notebooks/eda.ipynb` — label balance, feature sanity checks, graph statistics,
  caption uniqueness / leakage check
- `notebooks/demo_context.ipynb` — end-to-end inference on one raw audio file:
  waveform → graph → audio-derived caption → fused GNN-BERT prediction

## 6. Known limitations (also discussed in the report)

- GTZAN is single-label; the "multi-label" machinery (BCE, per-tag F1) is exercised
  with one positive tag per track. MagnaTagATune's top-50 tags would make it truly
  multi-label without code changes.
- Chord estimation is naive template matching over chroma, not a trained
  chord-recognition model.
- No artist metadata in GTZAN → possible residual artist overlap across splits;
  GTZAN's known duplicates/mislabelings apply.
- Captions describe audio, but through hand-designed descriptors; MusicCaps would
  provide genuinely independent human language (and is the intended Task 4 dataset).
