from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Output dataclass
@dataclass(frozen=True)
class AggregatedAudio:

    audio_gate: float
    noise_prob: float
    gasoline_prob: float
    cnn_confidence: float
    cnn_probabilities: np.ndarray
    entropy: float
    audio_label: str
    num_windows: int
    num_inlier_windows: int


# Aggregation function


def aggregate_audio_windows(results: Sequence) -> AggregatedAudio:
    
    results_list = list(results)

    if not results_list:
        return AggregatedAudio(
            audio_gate=0.0,
            noise_prob=0.0,
            gasoline_prob=0.0,
            cnn_confidence=0.0,
            cnn_probabilities=np.zeros(2, dtype=np.float32),
            entropy=1.0,
            audio_label="missing",
            num_windows=0,
            num_inlier_windows=0,
        )

    inlier_confidences: list[float] = []
    inlier_noise_probs: list[float] = []
    inlier_gasoline_probs: list[float] = []
    inlier_probabilities: list[np.ndarray] = []
    inlier_entropies: list[float] = []
    inlier_count = 0

    for item in results_list:
        if getattr(item, "ocsvm_label", None) != "inlier":
            continue

        inlier_count += 1

        # CNN confidence
        cnn_conf = getattr(item, "cnn_confidence", None)
        if cnn_conf is not None:
            try:
                inlier_confidences.append(float(cnn_conf))
            except (TypeError, ValueError):
                pass

        # Explicit named probabilities (preferred for gas-vs-noise models)
        noise_prob = getattr(item, "noise_prob", None)
        if noise_prob is not None:
            try:
                inlier_noise_probs.append(float(noise_prob))
            except (TypeError, ValueError):
                pass

        gasoline_prob = getattr(item, "gasoline_prob", None)
        if gasoline_prob is not None:
            try:
                inlier_gasoline_probs.append(float(gasoline_prob))
            except (TypeError, ValueError):
                pass

        # Per-class probabilities
        probs = getattr(item, "cnn_probabilities", None)
        if probs is not None:
            try:
                arr = np.asarray(probs, dtype=np.float32)
                if arr.ndim == 1 and arr.size > 0:
                    inlier_probabilities.append(arr)
            except (TypeError, ValueError):
                pass

        # Entropy
        ent = getattr(item, "entropy", None)
        if ent is not None:
            try:
                inlier_entropies.append(float(ent))
            except (TypeError, ValueError):
                pass

    # ── Aggregate ────────────────────────────────────────────────────────────
    cnn_confidence = float(np.mean(inlier_confidences)) if inlier_confidences else 0.0
    noise_prob = float(np.mean(inlier_noise_probs)) if inlier_noise_probs else 0.0
    gasoline_prob = (
        float(np.mean(inlier_gasoline_probs)) if inlier_gasoline_probs else 0.0
    )
    entropy = float(np.mean(inlier_entropies)) if inlier_entropies else 1.0

    if inlier_probabilities:
        # Stack into (n_inlier, num_classes) and mean across windows
        stacked = np.stack(inlier_probabilities, axis=0)
        cnn_probabilities = np.mean(stacked, axis=0).astype(np.float32)
    else:
        # Uniform distribution — represents total uncertainty
        num_classes = 2
        cnn_probabilities = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

    # Keep explicit noise/gasoline means synchronized with the vector summary.
    if cnn_probabilities.size >= 2:
        if inlier_noise_probs:
            cnn_probabilities[0] = np.float32(np.clip(noise_prob, 0.0, 1.0))
        if inlier_gasoline_probs:
            cnn_probabilities[1] = np.float32(np.clip(gasoline_prob, 0.0, 1.0))

    if inlier_count > 0:
        audio_label = "inlier"
        audio_gate = 1.0
    else:
        audio_label = "outlier"
        audio_gate = 0.0

    return AggregatedAudio(
        audio_gate=audio_gate,
        noise_prob=noise_prob,
        gasoline_prob=gasoline_prob,
        cnn_confidence=cnn_confidence,
        cnn_probabilities=cnn_probabilities,
        entropy=entropy,
        audio_label=audio_label,
        num_windows=len(results_list),
        num_inlier_windows=inlier_count,
    )
