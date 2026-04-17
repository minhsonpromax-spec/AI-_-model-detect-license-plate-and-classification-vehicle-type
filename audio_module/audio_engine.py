from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from .aggregation.audio_aggregation import (
    AggregatedAudio,
    aggregate_audio_windows,
)
from .config.config import AudioModuleConfig
from .fusion.fusion_engine import FusionEngine, FusionResult
from .inference.audio_pipeline import AudioInferenceResult, AudioPipeline


class AudioEngine:

    def __init__(
        self,
        config: AudioModuleConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        config.validate()
        self._cfg = config
        self._logger = logger or logging.getLogger(__name__)
        self._pipeline = AudioPipeline(config)
        self._fusion = FusionEngine(config)

    # gpt gợi ý chỗ này hay (factory helper)
    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> "AudioEngine":
        """Create an AudioEngine from a YAML config file."""
        cfg = AudioModuleConfig.from_yaml(config_path)
        return cls(cfg, logger=logger)
 


    def infer_audio_event(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        event_timestamp: float = 0.0,
    ) -> AggregatedAudio:

        samples = self._to_mono_float32(waveform) # nếu waveform là multi-channel, chuyển thành mono

        if samples.size == 0:
            self._logger.warning("[AudioEngine] Empty waveform received.")
            return aggregate_audio_windows([])

        windows = self._slice_windows(samples, sample_rate, event_timestamp)

        results: List[AudioInferenceResult] = []
        for w_start, w_end, w_samples in windows:
            try:
                result = self._pipeline.infer(
                    waveform=w_samples,
                    sample_rate=sample_rate,
                    window_start=w_start,
                    window_end=w_end,
                    logger=self._logger,
                )
                results.append(result)
                self._logger.debug(
                    "[AudioEngine] window [%.2f, %.2f] → ocsvm=%s cnn_conf=%.3f entropy=%.3f",
                    w_start, w_end, result.ocsvm_label,
                    result.cnn_confidence, result.entropy,
                ) # dùng %.f để đặt chỗ
            except Exception as exc:
                self._logger.error(
                    "[AudioEngine] Inference failed for window [%.2f, %.2f]: %s",
                    w_start, w_end, exc,
                )

        # Lưu lại để stats collector trong main.py đọc được
        self._last_window_results: List[AudioInferenceResult] = results
        return aggregate_audio_windows(results)


    def fuse_with_vision(
        self,
        vision_summary: Dict[str, float], # dict thật ra giống unordered_map hơn là map trong C++
        audio_result: AggregatedAudio,  # được trả về ở hàm infer_audio_event() ở trên
    ) -> dict:
        result: FusionResult = self._fusion.fuse(vision_summary, audio_result)
        self._logger.debug(
            "[AudioEngine] fusion → label=%s gas=%.3f elec=%.3f gate=%.1f H=%.3f",
            result.final_label, result.score_gasoline, result.score_electric,
            result.audio_gate, result.entropy,
        )
        return result.to_dict()

    # kết hợp infer và fusion
    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        vision_summary: Dict[str, float],
        event_timestamp: float = 0.0,
    ) -> dict:
        audio_result = self.infer_audio_event(waveform, sample_rate, event_timestamp)
        return self.fuse_with_vision(vision_summary, audio_result)


    # Internal helpers
    def _slice_windows(
        self,
        samples: np.ndarray,
        sample_rate: int,
        anchor_ts: float, # mốc thời gian
    ) -> List[tuple[float, float, np.ndarray]]:
        dur = self._cfg.window_duration
        stride = self._cfg.window_stride
        window_samples_count = int(dur * sample_rate)
        stride_samples_count = int(stride * sample_rate)

        total_samples = len(samples)
        windows: List[tuple[float, float, np.ndarray]] = []

        # Walk backward from the end of the array
        end_idx = total_samples
        while end_idx > 0:
            start_idx = max(0, end_idx - window_samples_count)
            chunk = samples[start_idx:end_idx]

            if chunk.size == 0:
                break

            # Timestamps relative to anchor_ts
            end_ts = anchor_ts - (total_samples - end_idx) / float(sample_rate)
            start_ts = end_ts - len(chunk) / float(sample_rate)

            windows.append((start_ts, end_ts, chunk))
            end_idx -= stride_samples_count

        # Return in chronological order (oldest first)
        windows.reverse()
        return windows

    @staticmethod # khởi truyền self
    def _to_mono_float32(waveform: np.ndarray) -> np.ndarray:
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=1)
        elif arr.ndim > 2:
            arr = arr.flatten()
        return arr
