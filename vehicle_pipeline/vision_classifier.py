"""
Interface phân loại xăng/điện từ hình ảnh đuôi xe (Vision).

Cách dùng:
    - Hiện tại: dùng PlaceholderVisionClassifier (return 50/50 uncertainty)
    - Khi có model thật: implement RealVisionClassifier, load model vào,
      rồi thay PlaceholderVisionClassifier bằng RealVisionClassifier trong main.py.

Contract của predict():
    Input  : numpy array (H, W, 3) — BGR frame từ OpenCV
    Output : dict {"gasoline": float, "electric": float}  — tổng = 1.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BaseVehicleTypeVision(ABC):
    """Abstract interface. Mọi implementation đều phải implement predict()."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> dict[str, float]:
        """
        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            {"gasoline": float, "electric": float} — tổng xấp xỉ 1.0.
        """
        ...


# ── Placeholder (dùng ngay, không cần model) ─────────────────────────────────

class PlaceholderVisionClassifier(BaseVehicleTypeVision):
    """
    Trả về 0.5 / 0.5 cho đến khi có model thật.
    FusionEngine sẽ dựa nhiều hơn vào audio trong trường hợp này.
    """

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        return {"gasoline": 0.5, "electric": 0.5}


# ── Real implementation (plug in khi có model) ────────────────────────────────

class RealVisionClassifier(BaseVehicleTypeVision):
    """
    Dùng model detect_type_vehicle.pt (YOLO classify) để phân loại xăng/điện.

    Model này nhận vào 1 frame ảnh, trả về xác suất của từng class.
    Giả định class 0 = gasoline, class 1 = electric (theo thứ tự training).
    Nếu ngược lại thì đổi chỉ số gas_idx / elec_idx bên dưới.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vision model không tìm thấy: {path}"
            )

        from ultralytics import YOLO
        self._model = YOLO(model_path)
        logger.info("[VisionClassifier] Loaded model: %s", model_path)

        # Chỉ số class trong model — đổi nếu model train theo thứ tự khác
        self._gas_idx = 0   # class 0 = gasoline
        self._elec_idx = 1  # class 1 = electric

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        # Chạy YOLO classify, verbose=False để không in log thừa ra terminal
        results = self._model(frame, verbose=False)[0]

        # results.probs.data là tensor xác suất của từng class
        probs = results.probs.data.cpu().numpy()

        # Đảm bảo không bị lỗi nếu model có ít class hơn mong đợi
        gas_prob  = float(probs[self._gas_idx])  if self._gas_idx  < len(probs) else 0.5
        elec_prob = float(probs[self._elec_idx]) if self._elec_idx < len(probs) else 0.5

        return {"gasoline": gas_prob, "electric": elec_prob}
