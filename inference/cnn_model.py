from __future__ import annotations

from pathlib import Path

import numpy as np


def _build_spatial_attention_block(): # gpt đề xuất cách này hay
    try:
        import tensorflow as tf
        from keras import layers
        from keras.utils import register_keras_serializable
    except ImportError as exc:
        raise ImportError(
            "TensorFlow / Keras is required "
            "pip install tensorflow"
        ) from exc

    @register_keras_serializable(package="CustomLayers")
    class SpatialAttentionBlock(layers.Layer):

        def __init__(self, kernel_size: int = 7, **kwargs):
            super().__init__(**kwargs)
            self.kernel_size = kernel_size
            self.conv = layers.Conv2D(
                1, (kernel_size, kernel_size),
                padding="same", activation="sigmoid",
            )

        def call(self, inputs):
            avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
            concat = tf.concat([avg_pool, max_pool], axis=-1)
            attention = self.conv(concat)
            return inputs * attention

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"kernel_size": self.kernel_size})
            return cfg

    return SpatialAttentionBlock


# CNNModel


class CNNModel:

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"CNN model not found: {path}")

        try:
            from keras.models import load_model
        except ImportError as exc:
            raise ImportError(
                "TensorFlow / Keras is required: pip install tensorflow"
            ) from exc

        SpatialAttentionBlock = _build_spatial_attention_block()
        self._model = load_model(
            path,
            custom_objects={"SpatialAttentionBlock": SpatialAttentionBlock},
        )

    def predict(self, pcen_spectrogram: np.ndarray) -> np.ndarray:

        # Shape: (1, n_mels, target_frames, 1)
        model_input = np.expand_dims(pcen_spectrogram, axis=(0, -1))
        raw = self._model.predict(model_input, verbose=0)
        probs = np.asarray(raw).flatten().astype(np.float32)

        # Binary sigmoid model (output shape = (1,)):
        # Giá trị duy nhất là xác suất của class gasoline (index 1)
        # → chuyển thành [noise_prob, gasoline_prob]
        if probs.size == 1:
            gas = float(np.clip(probs[0], 0.0, 1.0))
            return np.array([1.0 - gas, gas], dtype=np.float32)

        # Multi-class softmax: normalize để đảm bảo tổng = 1
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            num_classes = probs.shape[0]
            probs = np.ones(num_classes, dtype=np.float32) / float(num_classes)
        else:
            probs = probs / total

        return probs
