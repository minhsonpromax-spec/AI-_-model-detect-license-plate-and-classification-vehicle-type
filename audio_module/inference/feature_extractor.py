"""Trích xuất 43 đặc trưng dùng cho OCSVM"""
from __future__ import annotations

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise ImportError(
        "librosa is required for feature extraction: pip install librosa"
    ) from exc


_N_FFT = 1024
_HOP_LENGTH = 256
_N_MFCC = 13
_N_CHROMA = 12
_EXPECTED_DIM = 43  # 13 + 13 + 12 + 5


def extract_ocsvm_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:

    samples = np.array(samples, dtype=np.float32).copy()

    if samples.size == 0:
        raise ValueError("Audio samples must not be empty.")

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    # Normalize to peak amplitude (matches training pipeline)
    samples = librosa.util.normalize(samples)

    # MFCC (13) — mean across time frames
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sample_rate,
        n_mfcc=_N_MFCC, n_fft=_N_FFT, hop_length=_HOP_LENGTH,
    )
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta-MFCC (13) — mean across time frames
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Chroma STFT (12) — mean across time frames
    chroma = librosa.feature.chroma_stft(
        y=samples, sr=sample_rate,
        n_chroma=_N_CHROMA, n_fft=_N_FFT, hop_length=_HOP_LENGTH,
    )
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral features (5)
    spectral_features = np.array([
        float(np.mean(librosa.feature.spectral_centroid(
            y=samples, sr=sample_rate, n_fft=_N_FFT, hop_length=_HOP_LENGTH))),
        float(np.mean(librosa.feature.spectral_bandwidth(
            y=samples, sr=sample_rate, n_fft=_N_FFT, hop_length=_HOP_LENGTH))),
        float(np.mean(librosa.feature.spectral_rolloff(
            y=samples, sr=sample_rate, n_fft=_N_FFT, hop_length=_HOP_LENGTH))),
        float(np.mean(librosa.feature.zero_crossing_rate(y=samples))),
        float(np.mean(librosa.feature.rms(y=samples))),
    ], dtype=np.float32)

    feature_vector = np.concatenate(
        [mfcc_mean, delta_mean, chroma_mean, spectral_features]
    ).astype(np.float32)

    if feature_vector.size != _EXPECTED_DIM:
        raise ValueError(
            f"Feature extraction produced {feature_vector.size} dimensions; "
            f"expected {_EXPECTED_DIM}."
        )

    return feature_vector
