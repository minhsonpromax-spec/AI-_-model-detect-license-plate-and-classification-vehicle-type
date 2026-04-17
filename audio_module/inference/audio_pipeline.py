from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np


import librosa


try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None  # type: ignore[assignment]

from .feature_extractor import extract_ocsvm_features
from .ocsvm_model import OCSVMModel
from .cnn_model import CNNModel
from ..config.config import AudioModuleConfig


@dataclass(frozen=True)
class AudioInferenceResult: 
    # kết quả dự đoán của một cửa sổ âm thanh
    ocsvm_label: str
    cnn_probabilities: np.ndarray
    noise_prob: float
    gasoline_prob: float
    cnn_confidence: float
    entropy: float
    window_start: float = 0.0
    window_end: float = 0.0


# AudioPipeline

class AudioPipeline: # yêu cầu truyền instance của AudioModuleConfig vào

    def __init__(self, config: AudioModuleConfig) -> None:
        self._cfg = config
        self._ocsvm = OCSVMModel(config.ocsvm_model_path)
        self._cnn = CNNModel(config.cnn_model_path)

    def infer(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        window_start: float = 0.0,
        window_end: float = 0.0,
        logger: logging.Logger | None = None,
    ) -> AudioInferenceResult:

        samples = self._prepare(waveform, sample_rate, window_start, window_end, logger)

        # Stage 1: OCSVM gate 
        features = extract_ocsvm_features(samples, self._cfg.sample_rate)
        ocsvm_label, ocsvm_score = self._ocsvm.predict_gate(features)

        if logger:
            logger.debug(
                "[OCSVM] window [%.2f, %.2f] score=%.4f → %s%s",
                window_start, window_end, ocsvm_score, ocsvm_label,
                " (BYPASSED)" if self._cfg.bypass_ocsvm else "",
            )

        if ocsvm_label == "outlier" and not self._cfg.bypass_ocsvm:
            num_classes = self._cnn_num_classes()
            return AudioInferenceResult(
                ocsvm_label="outlier",
                cnn_probabilities=np.zeros(num_classes, dtype=np.float32),
                noise_prob=0.0,
                gasoline_prob=0.0,
                cnn_confidence=0.0,
                entropy=1.0,
                window_start=window_start,
                window_end=window_end,
            )

        # Stage 2: CNN classification 
        pcen = self._build_pcen(samples)
        probs = self._cnn.predict(pcen)
        noise_prob, gasoline_prob = self._named_probs(probs)

        entropy = self._normalized_entropy(probs)
        confidence = float(np.max(probs))

        return AudioInferenceResult(
            ocsvm_label="inlier",
            cnn_probabilities=probs,
            noise_prob=noise_prob,
            gasoline_prob=gasoline_prob,
            cnn_confidence=confidence,
            entropy=entropy,
            window_start=window_start,
            window_end=window_end,
        )

    # helper functions

    def _prepare(
        self,
        waveform: np.ndarray,
        source_sr: int,
        window_start: float,
        window_end: float,
        logger: logging.Logger | None,
    ) -> np.ndarray:
        """Flatten and resample to target sample rate if necessary."""
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim != 1:
            samples = samples.flatten()

        if source_sr == self._cfg.sample_rate:
            return samples

        try:
            samples = librosa.resample(
                samples, orig_sr=source_sr, target_sr=self._cfg.sample_rate
            )
            if logger:
                logger.debug(
                    "[AudioPipeline] Resampled %dHz → %dHz for window [%.2f, %.2f]",
                    source_sr, self._cfg.sample_rate, window_start, window_end,
                )
            return samples.astype(np.float32)
        except Exception as primary_err:
            if logger:
                logger.warning(
                    "[AudioPipeline] librosa resample failed (%s); trying scipy.",
                    primary_err,
                )

        # Fallback: scipy
        if scipy_signal is not None:
            try:
                n_target = int(len(samples) * self._cfg.sample_rate / source_sr)
                samples = scipy_signal.resample(samples, n_target)
                if logger:
                    logger.debug(
                        "[AudioPipeline] scipy fallback resample succeeded "
                        "for window [%.2f, %.2f].", window_start, window_end,
                    )
                return samples.astype(np.float32)
            except Exception as fallback_err:
                if logger:
                    logger.error(
                        "[AudioPipeline] scipy resample also failed (%s). "
                        "Proceeding at original rate — results may degrade.",
                        fallback_err,
                    )

        return samples

    def _build_pcen(self, samples: np.ndarray) -> np.ndarray:
        """Build fixed-size PCEN spectrogram for CNN input."""
        cfg = self._cfg
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=cfg.sample_rate,
            n_mels=cfg.n_mels,
            fmax=cfg.fmax,
            hop_length=cfg.hop_length,
            power=1.0,
        )
        mel = np.abs(mel).astype(np.float32) * float(2 ** 31)

        pcen = librosa.pcen(
            mel,
            sr=cfg.sample_rate,
            hop_length=cfg.hop_length,
            gain=0.98,
            bias=2.0,
            power=0.5,
            time_constant=0.4,
        )

        # Pad or trim to target_frames
        if pcen.shape[1] < cfg.target_frames:
            pad = cfg.target_frames - pcen.shape[1]
            pcen = np.pad(pcen, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
        elif pcen.shape[1] > cfg.target_frames:
            pcen = pcen[:, : cfg.target_frames]

        return pcen.astype(np.float32)

    @staticmethod
    def _normalized_entropy(probs: np.ndarray) -> float: # normalize entropy về [0, 1]
        n = len(probs)
        if n <= 1:
            return 0.0
        raw = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        return float(raw / np.log(n))

    def _cnn_num_classes(self) -> int:
        return len(self._cfg.class_label_map) or 2

    def _named_probs(self, probs: np.ndarray) -> tuple[float, float]:
        if probs.size == 0:
            return 0.0, 0.0

        label_to_idx = {v: k for k, v in self._cfg.class_label_map.items()}
        noise_idx = label_to_idx.get("noise", 0)
        gasoline_idx = label_to_idx.get("gasoline", 1)

        if noise_idx >= probs.size or gasoline_idx >= probs.size:
            return 0.0, 0.0

        noise_prob = float(np.clip(probs[noise_idx], 0.0, 1.0))
        gasoline_prob = float(np.clip(probs[gasoline_idx], 0.0, 1.0))
        return noise_prob, gasoline_prob
