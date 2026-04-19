"""
Load from YAML with AudioModuleConfig.from_yaml(path), or instantiate directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AudioModuleConfig:
    # Địa chỉ model
    ocsvm_model_path: str = "models/ocsvm_pipeline.pkl"
    cnn_model_path: str = "models/cnn_classifier.h5"

    # Thông số cấu hình xử lý âm thanh đầu vào
    sample_rate: int = 16000          
    fmax: int = 8000                  
    hop_length: int = 256             
    n_mels: int = 128                 
    target_frames: int = 64          

    # Cửa sổ trượt
    window_duration: float = 2.0      # mỗi cửa sổ trượt dài 2.0 giây
    window_stride: float = 1.0        # khoảng cách giữa hai cửa sổ trượt 1.0 giây

    # CNN  mapping 
    class_label_map: Dict[int, str] = field(
        default_factory=lambda: {0: "noise", 1: "gasoline"}
    )

    # Trọng số fusion (ưu tiên vision hay audio hơn)
    w_vision: float = 0.6            # trọng số ưu tiên vision
    w_audio: float = 0.4            # trọng số ưu tiên audio

    # Ngưỡng quyết định (nếu |score_gasoline - score_electric| < delta → "uncertain")
    delta: float = 0.20

    # Debug: bỏ qua OCSVM gate, cho CNN chạy trực tiếp trên mọi cửa sổ
    bypass_ocsvm: bool = False               

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AudioModuleConfig":
        """Load config from a YAML file. Requires PyYAML."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config loading: pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        if "class_label_map" in data:
            data["class_label_map"] = {
                int(k): str(v) for k, v in data["class_label_map"].items()
            }

        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def validate(self) -> None:
        """Raise ValueError if config is internally inconsistent."""
        label_values = set(self.class_label_map.values())
        for required in ("gasoline", "noise"):
            if required not in label_values:
                raise ValueError(
                    f"class_label_map must map some index to '{required}'. "
                    f"Got values: {label_values}"
                )
        if not (0.0 <= self.w_vision <= 1.0):
            raise ValueError(f"w_vision must be in [0, 1], got {self.w_vision}")
        if not (0.0 <= self.w_audio <= 1.0):
            raise ValueError(f"w_audio must be in [0, 1], got {self.w_audio}")
        if self.delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.window_duration <= 0.0:
            raise ValueError(f"window_duration must be > 0, got {self.window_duration}")
