from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import joblib
except ImportError as exc:
    raise ImportError(
        "joblib is required for OCSVM loading: pip install joblib"
    ) from exc


class OCSVMModel:

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"OCSVM model not found: {path}")
        self._model = joblib.load(path)

    def predict_gate(self, feature_vector: np.ndarray) -> tuple[str, float]:

        X = feature_vector.reshape(1, -1)
        score = float(self._model.decision_function(X)[0])
        label = "inlier" if score > 0.0 else "outlier"
        return label, score
