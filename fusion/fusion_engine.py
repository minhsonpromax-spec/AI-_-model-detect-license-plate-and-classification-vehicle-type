"""
Two-class fusion engine: combines vision and audio signals into a
final "gasoline" / "electric" / "uncertain" decision.

Formula  (Bayesian-Inspired Dynamic Weighting)
----------------------------------------------
    w_A_eff   = w_A * (1 - H^2) * gate
    S_gasoline = (w_V * V_gasoline + w_A_eff * A_gasoline) / (w_V + w_A_eff)
    S_electric = (w_V * V_electric + w_A_eff * A_noise)    / (w_V + w_A_eff)

Where:
    V_*    — vision score for class (from vision_summary dict)
    A_gasoline — CNN gasoline probability (from AggregatedAudio)
    A_noise    — CNN noise/non-gasoline probability (from AggregatedAudio)
    H      — normalized audio entropy  [0, 1]
    gate   — audio_gate  ∈ {0.0, 1.0}
    w_V    — vision weight  (from config)
    w_A    — audio weight   (from config)

Rationale:
* Quadratic entropy penalty (1 - H^2) is gentler for mild noise and only
  punishes heavily when entropy is truly high (Bayesian-inspired weighting).
* (1 - H^2) is applied to BOTH numerator and denominator so the score
  tracks the more confident modality instead of being averaged down.
* Denominator is always >= w_V > 0 (safety: never division by zero).

Decision rule:
    if |S_gasoline - S_electric| < delta  →  "uncertain"
    else                                  →  argmax(S_gasoline, S_electric)

Constraints enforced
--------------------
* OCSVM score is NEVER used here.
* gate is the ONLY binary switch; entropy modulates audio contribution.
* Requires class_label_map to contain "gasoline" and "noise".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from aggregation.audio_aggregation import AggregatedAudio
from config.config import AudioModuleConfig


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionResult:
    """
    Per-event fusion output.

    Fields
    ------
    final_label   : "gasoline" | "electric" | "uncertain"
    score_gasoline: Fusion score for gasoline class.
    score_electric: Fusion score for electric class.
    audio_gate    : 1.0 if audio was usable (any inlier), else 0.0.
    entropy       : Mean audio entropy from inlier windows (1.0 if no audio).
    """

    final_label: str
    score_gasoline: float
    score_electric: float
    audio_gate: float
    entropy: float

    def to_dict(self) -> dict:
        return {
            "final_label": self.final_label,
            "score_gasoline": float(self.score_gasoline),
            "score_electric": float(self.score_electric),
            "audio_gate": float(self.audio_gate),
            "entropy": float(self.entropy),
        }


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Stateless, deterministic two-class fusion engine.

    Args:
        config: AudioModuleConfig.  Must pass config.validate() before use.
    """

    def __init__(self, config: AudioModuleConfig) -> None:
        config.validate()
        self._cfg = config
        # Build reverse lookup: class_name → index in CNN probability vector
        self._label_to_idx: Dict[str, int] = {
            v: k for k, v in config.class_label_map.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        vision_summary: Dict[str, float],
        audio_result: AggregatedAudio,
    ) -> FusionResult:
        """
        Compute fusion scores and determine final vehicle fuel type.

        Args:
            vision_summary : dict with keys "gasoline" and "electric"
                             (float scores in [0, 1]).
            audio_result   : AggregatedAudio from aggregate_audio_windows().

        Returns:
            FusionResult

        Raises:
            KeyError:  if vision_summary is missing "gasoline" or "electric".
            ValueError: if audio result has unexpected probability dimensions.
        """
        self._validate_vision_summary(vision_summary)

        V_gasoline = float(np.clip(vision_summary["gasoline"], 0.0, 1.0))
        V_electric = float(np.clip(vision_summary["electric"], 0.0, 1.0))

        gate = float(audio_result.audio_gate)          # ∈ {0.0, 1.0}
        H = float(np.clip(audio_result.entropy, 0.0, 1.0))

        A_gasoline, A_noise = self._extract_audio_class_scores(audio_result)

        w_V = self._cfg.w_vision
        w_A = self._cfg.w_audio

        # ── Core fusion formula (Bayesian-Inspired Dynamic Weighting) ────────
        # Quadratic entropy penalty: (1 - H^2) is lenient on mild noise but
        # collapses to 0 only when entropy saturates. Applied symmetrically
        # to numerator and denominator so the score follows the more
        # confident modality instead of being averaged down.
        w_A_eff = w_A * (1.0 - H * H) * gate
        denom = w_V + w_A_eff  # >= w_V > 0  (safety: never zero)

        S_gasoline = (w_V * V_gasoline + w_A_eff * A_gasoline) / denom
        S_electric = (w_V * V_electric + w_A_eff * A_noise) / denom

        # ── Decision ─────────────────────────────────────────────────────────
        gap = abs(S_gasoline - S_electric)
        if gap < self._cfg.delta:
            final_label = "uncertain"
        elif S_gasoline > S_electric:
            final_label = "gasoline"
        else:
            final_label = "electric"

        return FusionResult(
            final_label=final_label,
            score_gasoline=float(S_gasoline),
            score_electric=float(S_electric),
            audio_gate=gate,
            entropy=H,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_audio_class_scores(
        self, audio_result: AggregatedAudio
    ) -> tuple[float, float]:
        """
        Map CNN probability vector to (A_gasoline, A_noise).

        Returns (0.5, 0.5) if probabilities vector is empty or indices are missing,
        so the audio term contributes equal weight rather than corrupting fusion.
        """
        probs = audio_result.cnn_probabilities
        if probs is None or probs.size == 0:
            return 0.5, 0.5

        gas_idx = self._label_to_idx.get("gasoline")
        noise_idx = self._label_to_idx.get("noise")

        if gas_idx is None or noise_idx is None:
            return 0.5, 0.5

        if gas_idx >= probs.size or noise_idx >= probs.size:
            raise ValueError(
                f"CNN probability vector has {probs.size} elements, but "
                f"class_label_map references indices {gas_idx} (gasoline) "
                f"and {noise_idx} (noise). Update class_label_map in config."
            )

        A_gasoline = float(np.clip(probs[gas_idx], 0.0, 1.0))
        A_noise = float(np.clip(probs[noise_idx], 0.0, 1.0))
        return A_gasoline, A_noise

    @staticmethod
    def _validate_vision_summary(vision_summary: Dict[str, float]) -> None:
        for key in ("gasoline", "electric"):
            if key not in vision_summary:
                raise KeyError(
                    f"vision_summary must contain key '{key}'. "
                    f"Got keys: {list(vision_summary.keys())}"
                )
