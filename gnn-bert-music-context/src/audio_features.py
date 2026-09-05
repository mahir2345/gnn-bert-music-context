"""
audio_features.py
==================
Extracts mel-spectrogram / chroma features from raw audio and splits
tracks into fixed-length segments for downstream graph construction.

Used by: graph_builder.py, eda.ipynb
"""

import numpy as np
import librosa


def load_audio(path: str, sample_rate: int = 22050) -> np.ndarray:
    """Load an audio file and resample to the target sample rate (mono)."""
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y


def extract_log_mel(y: np.ndarray, sr: int = 22050, n_mels: int = 128) -> np.ndarray:
    """
    Compute a log-scaled mel spectrogram.

    Returns
    -------
    np.ndarray of shape (n_mels, n_frames)
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return _normalize(log_mel)


def extract_log_mel_pooled(
    y: np.ndarray, sr: int = 22050, n_mels: int = 128, time_pool: int = 4
) -> np.ndarray:
    """
    Log-mel spectrogram average-pooled along time by `time_pool` and stored
    as float16 -- the compact on-disk format used for the CNN baseline
    (data/processed/mel/<track_id>.npy). A 30 s GTZAN clip becomes
    (128, ~323) instead of (128, ~1292), which is 8x smaller on disk.
    """
    mel = extract_log_mel(y, sr=sr, n_mels=n_mels)  # normalized (n_mels, T)
    t = (mel.shape[1] // time_pool) * time_pool
    pooled = mel[:, :t].reshape(mel.shape[0], -1, time_pool).mean(axis=2)
    return pooled.astype(np.float16)


def extract_chroma(y: np.ndarray, sr: int = 22050, n_chroma: int = 12) -> np.ndarray:
    """
    Compute a chromagram (pitch-class energy), useful for chord-transition graphs.

    Returns
    -------
    np.ndarray of shape (n_chroma, n_frames)
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma)
    return _normalize(chroma)


def _normalize(feat: np.ndarray) -> np.ndarray:
    """Per-track normalization: zero mean, unit variance (avoids div-by-zero)."""
    mean = feat.mean()
    std = feat.std()
    return (feat - mean) / (std + 1e-8)


def segment_track(
    y: np.ndarray,
    sr: int = 22050,
    segment_seconds: float = 5.0,
    hop_seconds: float = 5.0,
    beat_synchronous: bool = False,
):
    """
    Split a raw waveform into fixed-length (or beat-synchronous) segments.

    Returns
    -------
    List[np.ndarray]  -- list of waveform chunks, one per segment
    """
    if beat_synchronous:
        return _beat_synchronous_segments(y, sr)

    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)

    segments = []
    for start in range(0, max(len(y) - seg_len, 0) + 1, hop_len):
        segments.append(y[start:start + seg_len])

    # Always include a final short tail segment if meaningful audio remains
    if len(y) % hop_len > sr * 0.5:  # more than 0.5s of leftover audio
        segments.append(y[-seg_len:])

    return segments


def _beat_synchronous_segments(y: np.ndarray, sr: int):
    """Segment audio between detected beat frames instead of fixed windows."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_samples = librosa.frames_to_samples(beat_frames)
    boundaries = np.concatenate([[0], beat_samples, [len(y)]])

    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end > start:
            segments.append(y[start:end])
    return segments


def segment_embedding(segment: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Produce a single fixed-size feature vector for one segment
    (mean-pooled log-mel + chroma), used to initialize GNN node features h_i^(0).
    """
    if len(segment) < sr * 0.1:  # too short to featurize meaningfully
        segment = np.pad(segment, (0, int(sr * 0.1) - len(segment)))

    mel = extract_log_mel(segment, sr=sr)
    chroma = extract_chroma(segment, sr=sr)

    mel_vec = mel.mean(axis=1)       # (n_mels,)
    chroma_vec = chroma.mean(axis=1)  # (n_chroma,)

    return np.concatenate([mel_vec, chroma_vec])  # (n_mels + n_chroma,)


if __name__ == "__main__":
    # Quick smoke test with a synthetic sine wave (no real file needed)
    sr = 22050
    t = np.linspace(0, 10, sr * 10)
    y_test = 0.5 * np.sin(2 * np.pi * 440 * t)

    mel = extract_log_mel(y_test, sr=sr)
    chroma = extract_chroma(y_test, sr=sr)
    segments = segment_track(y_test, sr=sr, segment_seconds=5.0, hop_seconds=5.0)

    print(f"log-mel shape:  {mel.shape}")
    print(f"chroma shape:   {chroma.shape}")
    print(f"num segments:   {len(segments)}")
    print(f"segment embedding shape: {segment_embedding(segments[0], sr=sr).shape}")
