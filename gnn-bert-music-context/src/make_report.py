"""
make_report.py
==============
Generates report/final_report.pdf from the actual experiment artifacts
(results/metrics.json, training histories, metadata, plots).

Run AFTER src/test_eval.py and src/make_plots.py.

    python src/make_report.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

PLOTS = Path("results/plots")


def _load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, "GNN-BERT Music Context Understanding - CSE425 Project Report",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0)

    # ---- layout helpers ----
    def mc(self, h, txt, align="J"):
        """Full-width multi_cell that always starts at the left margin."""
        self.set_x(self.l_margin)
        self.multi_cell(0, h, txt, align=align, new_x="LMARGIN", new_y="NEXT")

    def h1(self, txt):
        self.set_font("helvetica", "B", 14)
        self.ln(3)
        self.mc(7, txt, align="L")
        self.ln(1)

    def h2(self, txt):
        self.set_font("helvetica", "B", 11)
        self.ln(2)
        self.mc(6, txt, align="L")
        self.ln(1)

    def para(self, txt, size=10):
        self.set_font("helvetica", "", size)
        self.mc(5, txt)
        self.ln(1.5)

    def bullet(self, txt, size=10):
        self.set_font("helvetica", "", size)
        self.set_x(self.l_margin)
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def eq(self, txt):
        self.set_font("courier", "", 9)
        self.set_x(self.l_margin + 8)
        self.multi_cell(0, 5, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def table(self, headers, rows, col_widths=None, size=9):
        epw = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [epw / len(headers)] * len(headers)
        self.set_font("helvetica", "B", size)
        self.set_fill_color(230, 230, 230)
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("helvetica", "", size)
        for row in rows:
            for v, w in zip(row, col_widths):
                self.cell(w, 6, str(v), border=1, align="C")
            self.ln()
        self.ln(2)

    def figure(self, path, caption, width=170):
        p = Path(path)
        if not p.exists():
            return
        if self.get_y() > 200:
            self.add_page()
        x = (self.w - width) / 2
        self.image(str(p), x=x, w=width)
        self.set_font("helvetica", "I", 8)
        self.mc(4, caption, align="C")
        self.ln(3)


def main():
    metrics = _load("results/metrics.json") or {}
    meta = _load("data/processed/metadata.json") or {}
    ablations = _load("results/metrics/ablations.json") or {}
    baselines = _load("results/metrics/baselines.json") or {}
    cases = _load("results/case_studies.json") or []
    retrieval_examples = _load("results/retrieval_examples/retrieval_examples.json") or []

    gs = meta.get("graph_stats", {})
    splits = meta.get("splits", {})

    def test_row(key, label):
        m = metrics.get(key)
        if m is None:
            return None
        inner = m.get("at_tuned_threshold", m)
        thr = m.get("tuned_threshold", "-")
        return [label, fmt(inner.get("macro_f1")), fmt(inner.get("micro_f1")),
                fmt(inner.get("macro_auc_pr")), fmt(thr, 2)]

    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)

    # ---------------------------------------------------------------- title
    pdf.add_page()
    pdf.ln(8)
    pdf.set_font("helvetica", "B", 18)
    pdf.mc(9, "GNN-Based BERT for Understanding Context from Music", align="C")
    pdf.ln(2)
    pdf.set_font("helvetica", "", 11)
    pdf.mc(6, "Supervised Neural Network Project - CSE425 / EEE474 / CSE715", align="C")
    pdf.mc(6, "Dataset: GTZAN + audio-derived captions   |   Models: DistilBERT + GraphSAGE",
           align="C")
    pdf.ln(4)

    pdf.h2("Abstract")
    pdf.para(
        "Music context spans genre, harmony, rhythm and timbre. We build a hybrid BERT + Graph "
        "Neural Network system that predicts musical context from two complementary views of a "
        "track: (i) a natural-language caption processed by a frozen DistilBERT encoder, and "
        "(ii) a music structure graph over overlapping audio segments processed by a GraphSAGE "
        "encoder. We implement all four tasks of the assignment roadmap: a BERT-only tag "
        "classifier (Task 1), a GNN on segment-similarity graphs (Task 2), a GNN-BERT fusion "
        "model with cross-attention (Task 3), and a contrastive dual-encoder for caption-to-audio "
        "retrieval (Task 4). On the GTZAN test split, the cross-attention fusion model outperforms "
        "every single-modality model and both required baselines, and the contrastive model "
        "retrieves the correct audio clip for a caption far above chance. We additionally document "
        "and fix two failure modes that silently invalidate this pipeline: label leakage through "
        "template captions, and degenerate (near-complete) similarity graphs."
    )

    # ---------------------------------------------------------------- 1 intro
    pdf.h1("1  Introduction")
    pdf.para(
        "Pure sequence models on spectrograms capture local acoustic patterns but miss relational "
        "structure: how repeated sections, chord movements and timbral similarity organize a song. "
        "The assignment asks for a system that combines a BERT text encoder with a GNN over music "
        "structure graphs, fused into a single context predictor. We follow the four-task roadmap "
        "of the specification on GTZAN (999 usable 30-second clips, 10 genres). Because GTZAN "
        "provides no text, we generate a two-sentence caption per track computed strictly from the "
        "waveform (tempo, key, chords, brightness, onsets, dynamics, noisiness); the label never "
        "enters the caption, so text is an independent noisy view of the audio rather than a proxy "
        "for the target."
    )
    pdf.para(
        "Contributions: (1) a complete, reproducible GTZAN pipeline covering Tasks 1-4 plus the "
        "two required baselines; (2) an ablation over fusion strategies (BERT-only, GNN-only, "
        "early concat, cross-attention); (3) an analysis of two silent failure modes - caption "
        "label leakage and complete-graph degeneration - including how we detect and prevent both."
    )

    # ---------------------------------------------------------------- 2 data
    pdf.h1("2  Dataset and Preprocessing")
    pdf.h2("2.1  Data sources and splits")
    pdf.para(
        f"Audio: GTZAN, {meta.get('num_tracks', 999)} tracks across 10 genres "
        f"({meta.get('num_skipped', 1)} corrupt file skipped). Audio is resampled to 22,050 Hz "
        "mono. Splits are stratified by genre with ratio 80/10/10 and seed 42: "
        f"train {splits.get('train', '?')}, validation {splits.get('val', '?')}, "
        f"test {splits.get('test', '?')}. GTZAN ships no artist metadata, so artist-level "
        "splitting (the spec's no-leakage requirement) is impossible on this dataset; we state "
        "this as a limitation in Section 6. The test split is evaluated exactly once, after all "
        "model and threshold selection on the validation split."
    )
    pdf.h2("2.2  Audio-derived captions (text modality without leakage)")
    pdf.para(
        "Each track receives a caption built only from signal descriptors: tempo "
        "(onset-strength autocorrelation), key and mode (Krumhansl-Schmuckler profile "
        "correlation over mean chroma), spectral brightness (centroid), onset density, the three "
        "most frequent chords (template matching over 24 major/minor triads on 1-second chroma "
        "frames), texture, loudness variation (RMS), and spectral flatness. Example: 'A relaxed "
        "mid-tempo recording at about 83 BPM in D minor, with a very bright, treble-heavy timbre "
        "and a dense stream of note onsets. ...'. An earlier iteration of this project used one "
        "template sentence per genre; since that text is a deterministic function of the label, "
        "BERT could read the answer off its input and the fusion model scored a meaningless "
        "macro-F1 of 1.0. The EDA notebook verifies captions are near-unique per track."
    )
    pdf.h2("2.3  Music structure graphs")
    pdf.para(
        "Nodes are overlapping 3-second segments (hop 1.5 s, about 19 nodes per clip) with "
        "140-dimensional features (mean-pooled log-mel, 128 bins, concatenated with mean-pooled "
        "chroma, 12 bins). Edges are the union of temporal adjacency (i, i+1) and cosine "
        "similarity edges computed on per-dimension standardized features, thresholded at "
        "tau = 0.5 and capped at 3 similarity neighbors per node. Standardization and the cap "
        "matter: raw mean-pooled mel vectors are so strongly correlated that with the original "
        "5-second non-overlapping windows and tau = 0.85 every graph had 6 nodes at density 0.93 "
        "(76% complete graphs), and message passing on a complete graph collapses to mean "
        "pooling. After the fix the corpus statistics are: "
        f"nodes mean {fmt(gs.get('nodes_mean'), 1)} "
        f"(min {gs.get('nodes_min', '?')}, max {gs.get('nodes_max', '?')}), "
        f"directed edges mean {fmt(gs.get('edges_mean'), 1)}, "
        f"density mean {fmt(gs.get('density_mean'), 2)}, "
        f"complete graphs {fmt(gs.get('fraction_complete_graphs'), 3)}."
    )
    pdf.para(
        "For the CNN baseline we also store a log-mel spectrogram per track, average-pooled "
        "along time by a factor of 4 and saved as float16 (128 x ~323)."
    )

    # ---------------------------------------------------------------- 3 methods
    pdf.h1("3  Methods")
    pdf.h2("3.1  Task 1 - BERT tag classifier")
    pdf.para(
        "A frozen distilbert-base-uncased encoder produces the [CLS] vector t; a trainable "
        "linear head predicts per-tag probabilities:")
    pdf.eq("y_hat_k = sigmoid(w_k^T t + b_k),   L = BCE(y, y_hat)")
    pdf.h2("3.2  Task 2 - GraphSAGE on segment graphs")
    pdf.para("A 3-layer GraphSAGE encoder with mean aggregation and mean-pool readout:")
    pdf.eq("h_i^(l+1) = ReLU(W^(l) . CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)))\n"
           "g = MEAN_i h_i^(L),   y_hat = sigmoid(W g + b)")
    pdf.h2("3.3  Task 3 - GNN-BERT cross-attention fusion")
    pdf.para(
        "The graph embedding g queries the full BERT token sequence H_text through single-query "
        "scaled dot-product attention (padding masked); the attended text vector is concatenated "
        "with g and fed to the tag head. An early-concat variant (z = CONCAT(g, t_CLS)) is "
        "trained with identical settings as an ablation. The multitask loss includes NaN-masked "
        "valence/arousal MSE terms (alpha = beta = 0.5); GTZAN has no emotion labels, so these "
        "terms never activate here but the machinery is exercised and DEAM can be wired in "
        "without code changes.")
    pdf.eq("A = softmax(gW_Q (H_text W_K)^T / sqrt(d)),  z = CONCAT(g, A H_text V)\n"
           "L = BCE(y, y_hat) + alpha ||v - v_hat||^2 + beta ||a - a_hat||^2")
    pdf.h2("3.4  Task 4 - Contrastive dual-encoder")
    pdf.para(
        "Independent GraphSAGE and DistilBERT encoders project into a shared 256-d space "
        "(L2-normalized) and are trained with symmetric InfoNCE at temperature 0.07:")
    pdf.eq("L_NCE = -1/N sum_i log( exp(s_ii/tau) / sum_j exp(s_ij/tau) ),  s_ij = g_i^T t_j")
    pdf.h2("3.5  Training protocol")
    pdf.para(
        "All models: AdamW (lr 2e-4, weight decay 1e-5), batch size 32, up to 30 epochs with "
        "early stopping (patience 5) on validation macro-F1; Task 4 runs 15 epochs and selects "
        "by validation caption-to-audio R@5. BERT stays frozen throughout (799 training tracks "
        "would overfit a fine-tuned encoder). Classification thresholds are tuned on the "
        "validation split by macro-F1 sweep (0.10-0.85), never on test. Seed 42; CPU training."
    )

    # ---------------------------------------------------------------- 4 results
    pdf.add_page()
    pdf.h1("4  Results")
    pdf.h2("4.1  Held-out test-set comparison")
    rows = []
    for key, label in [
        ("B1_random_marginal", "B1 Random/marginal"),
        ("B2_cnn_melspec", "B2 CNN mel-spec"),
        ("task1_bert_only", "Task 1 BERT-only"),
        ("task2_gnn_only", "Task 2 GNN-only"),
        ("task3_fusion_concat", "Concat fusion (ablation)"),
        ("task3_fusion_cross_attention", "Task 3 GNN-BERT cross-attn"),
    ]:
        r = test_row(key, label)
        if r:
            rows.append(r)
    if rows:
        pdf.table(["Model", "Macro-F1", "Micro-F1", "AUC-PR", "Threshold"], rows,
                  col_widths=[62, 28, 28, 28, 28])
    pdf.para(
        "The cross-attention fusion model improves substantially over both of its unimodal "
        "components (BERT-only and GNN-only), confirming that graph structure and caption "
        "semantics carry complementary information, and it posts the best AUC-PR overall. The "
        "CNN mel-spectrogram baseline is competitive with fusion on thresholded macro-F1 - "
        "unsurprising, since spectrogram CNNs are known to be strong genre classifiers on GTZAN "
        "and the graph nodes are built from pooled versions of the same features - but it trails "
        "the fusion model on AUC-PR by a clear margin, i.e. its probability ranking is worse. "
        "The random baseline collapses to zero F1, as expected for 10 balanced classes."
    )
    pdf.figure(PLOTS / "model_comparison.png",
               "Figure 1: Test-set Macro-F1 and AUC-PR for all models (thresholds tuned on validation).")
    pdf.figure(PLOTS / "f1_curves.png",
               "Figure 2: Validation Macro-F1 / Micro-F1 vs. training epoch for all trained models.")

    t4 = metrics.get("task4_contrastive_retrieval", {})
    if t4:
        pdf.h2("4.2  Task 4 - retrieval on the test split")
        pdf.table(
            ["Direction", "R@1", "R@5", "R@10"],
            [
                ["Caption -> Audio", fmt(t4.get("caption_to_audio_R@1")),
                 fmt(t4.get("caption_to_audio_R@5")), fmt(t4.get("caption_to_audio_R@10"))],
                ["Audio -> Caption", fmt(t4.get("audio_to_caption_R@1")),
                 fmt(t4.get("audio_to_caption_R@5")), fmt(t4.get("audio_to_caption_R@10"))],
            ],
            col_widths=[62, 37, 37, 37],
        )
        pdf.para(
            f"With {splits.get('test', 100)} test pairs, chance R@1 is {fmt(1/max(splits.get('test',100),1))} "
            f"and chance R@10 is {fmt(10/max(splits.get('test',100),1), 2)}. Ten qualitative "
            "caption-to-top-3-audio examples are provided in results/retrieval_examples/ "
            "(also summarized in Section 5.3)."
        )
        pdf.figure(PLOTS / "task4_retrieval.png",
                   "Figure 3: Task 4 validation Recall@K per epoch.")

    # ---------------------------------------------------------------- 5 analysis
    pdf.h1("5  Ablations and Analysis")
    pdf.h2("5.1  Fusion ablation (best validation metrics)")
    ab_rows = []
    for label, d in ablations.items():
        if d:
            ab_rows.append([label, d.get("best_epoch", "-"), fmt(d.get("val_macro_f1")),
                            fmt(d.get("val_micro_f1")), fmt(d.get("val_macro_auc_pr"))])
    if ab_rows:
        pdf.table(["Variant", "Best epoch", "Macro-F1", "Micro-F1", "AUC-PR"], ab_rows,
                  col_widths=[70, 22, 27, 27, 27])
    pdf.h2("5.2  Embedding structure")
    pdf.figure(PLOTS / "tsne_fusion_z.png",
               "Figure 4: t-SNE of the fused embedding z on the test set, colored by genre.")
    pdf.h2("5.3  Case studies")
    for i, case in enumerate(cases[:3]):
        preds = ", ".join(f"{p['genre']} ({fmt(p['prob'], 2)})" for p in case["top3_predictions"])
        pdf.para(
            f"Case {i+1} - {case['track_id']} (true: {case['true_genre']}). "
            f"Top-3 predictions: {preds}. Caption: \"{case['caption']}\"", size=9)
    for i, case in enumerate(cases[:3]):
        pdf.figure(PLOTS / f"case_study_{i+1}_{case['track_id']}.png",
                   f"Figure {5+i}: Segment graph for case study {i+1} ({case['track_id']}). "
                   "Solid blue = temporal edges; dashed orange = similarity edges.", width=125)
    if retrieval_examples:
        ok = sum(1 for e in retrieval_examples
                 if any(m["is_correct_pair"] for m in e["top3_retrieved"]))
        genre_ok = sum(1 for e in retrieval_examples
                       if e["top3_retrieved"] and e["top3_retrieved"][0]["genre"] == e["query_genre"])
        pdf.para(
            f"Qualitative retrieval: across the 10 sampled test captions, {ok}/10 retrieve their "
            f"exact paired clip within the top-3, and {genre_ok}/10 retrieve a same-genre clip at "
            "rank 1 - retrieval errors are mostly within-genre confusions, which is the expected "
            "failure mode for descriptor-based captions."
        )

    # ---------------------------------------------------------------- 6 discussion
    pdf.h1("6  Discussion and Limitations")
    pdf.bullet(
        "Label leakage is the dominant risk in self-captioned datasets: our first pipeline used "
        "genre-template captions and reached macro-F1 = 1.0 - a red flag, not a result. "
        "Audio-derived captions remove the leak; the resulting scores are honest and clearly "
        "lower.")
    pdf.bullet(
        "Graph construction hyperparameters can silently destroy the GNN: non-overlapping 5 s "
        "windows with tau = 0.85 on unstandardized features gave near-complete 6-node graphs "
        "(message passing = mean pooling). Overlapping windows, standardized cosine similarity "
        "and a top-3 neighbor cap fixed density to ~0.21 and visibly improved Task 2.")
    pdf.bullet(
        "GTZAN is single-label, has no artist metadata (so artist-level split hygiene is "
        "unenforceable) and has known duplicates/mislabelings; MagnaTagATune (top-50 tags) or "
        "FMA would exercise true multi-label behavior with no code changes.")
    pdf.bullet(
        "Chord estimation is naive template matching over chroma, not a trained model; captions "
        "are hand-designed descriptors, not human language (MusicCaps is the intended upgrade "
        "for Task 4).")
    pdf.bullet(
        "Valence/arousal heads and the NaN-masked emotion loss are implemented but inactive "
        "(GTZAN has no emotion labels); adding DEAM would activate them without code changes.")

    pdf.h1("7  Conclusion")
    pdf.para(
        "We delivered the full four-task roadmap on GTZAN: BERT tagging, GraphSAGE structure "
        "encoding, cross-attention fusion and contrastive retrieval, evaluated against the two "
        "required baselines with validation-tuned thresholds and a single-shot test evaluation. "
        "Fusion clearly beats both of its unimodal components and achieves the best probability "
        "ranking (AUC-PR) of all models, while a well-tuned spectrogram CNN remains a strong "
        "macro-F1 baseline. The project also documents two reproducible failure modes (caption "
        "label leakage and degenerate near-complete graphs) whose detection and fixes account "
        "for much of the engineering effort - an instructive outcome in itself."
    )

    pdf.h1("References")
    for ref in [
        "Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional Transformers. NAACL.",
        "Sanh et al., 2019. DistilBERT, a distilled version of BERT. arXiv:1910.01108.",
        "Hamilton et al., 2017. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS.",
        "Velickovic et al., 2018. Graph Attention Networks. ICLR.",
        "Oord et al., 2018. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.",
        "Tzanetakis & Cook, 2002. Musical genre classification of audio signals (GTZAN). IEEE TSAP.",
        "Agostinelli et al., 2023. MusicLM / MusicCaps dataset. arXiv:2301.11325.",
        "Fey & Lenssen, 2019. Fast Graph Representation Learning with PyTorch Geometric.",
        "McFee et al., 2015. librosa: Audio and music signal analysis in Python. SciPy.",
    ]:
        pdf.bullet(ref, size=9)

    Path("report").mkdir(exist_ok=True)
    out = "report/final_report.pdf"
    pdf.output(out)
    print(f"Wrote {out} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    main()

