# Chi tiết Audio Module — `audio_module/`

Module này xử lý **toàn bộ pipeline âm thanh** — từ waveform thô đến nhãn `gasoline`/`noise`.  
Được thiết kế độc lập, có thể import và dùng riêng mà không cần phần video pipeline.

```python
from audio_module import AudioEngine

engine = AudioEngine.from_yaml("audio_module/config/config.yaml")
result = engine.process(waveform, sample_rate=16000, vision_summary={...})
# result = {"final_label": "gasoline", "score_gasoline": 0.58, ...}
```

---

## Tổng quan các thành phần

```
AudioEngine  (audio_engine.py)
    │
    ├── AudioPipeline  (inference/audio_pipeline.py)
    │       │
    │       ├── feature_extractor.py  → 43 đặc trưng cho OCSVM
    │       ├── OCSVMModel            → gate (inlier / outlier)
    │       └── CNNModel              → gasoline_prob, entropy
    │
    ├── aggregate_audio_windows  (aggregation/audio_aggregation.py)
    │       → tổng hợp N windows → 1 AggregatedAudio
    │
    └── FusionEngine  (fusion/fusion_engine.py)
            → vision_summary + AggregatedAudio → FusionResult
```

---

## `audio_engine.py` — Bộ điều phối chính

`AudioEngine` là class duy nhất mà code bên ngoài cần biết.

### Khởi tạo

```python
# Từ file yaml (cách dùng trong production):
engine = AudioEngine.from_yaml("audio_module/config/config.yaml")

# Hoặc trực tiếp:
from audio_module.config.config import AudioModuleConfig
cfg = AudioModuleConfig(ocsvm_model_path="...", cnn_model_path="...")
engine = AudioEngine(cfg, logger=my_logger)
```

### Sliding window

Khi nhận waveform 4-5 giây, AudioEngine **không xử lý 1 lần** mà chia thành các **cửa sổ 2 giây** chồng lên nhau:

```
Waveform: |────────────────────────────|  (4.87s)
Window 1: |══════════|                     (-0.33 → 0.53s)
Window 2: |──|══════════|                  (-0.33 → 1.53s)
Window 3:    |──|══════════|               (0.53 → 2.53s)
Window 4:       |──|══════════|            (1.53 → 3.53s)
Window 5:          |──|══════════|         (2.53 → 4.53s)
          stride = 1s, window = 2s → 5 windows
```

```python
# Mỗi window được xử lý độc lập:
for w_start, w_end, w_samples in windows:
    result = self._pipeline.infer(w_samples, sample_rate, w_start, w_end)
    results.append(result)

# Lưu lại để stats collector đọc:
self._last_window_results = results

return aggregate_audio_windows(results)
```

### `process()` — API đơn giản nhất

```python
def process(self, waveform, sample_rate, vision_summary, event_timestamp=0.0) -> dict:
    audio_result = self.infer_audio_event(waveform, sample_rate, event_timestamp)
    return self.fuse_with_vision(vision_summary, audio_result)
```

Chỉ cần gọi `process()` — nó gọi cả 2 bước infer + fusion bên trong.

---

## `inference/audio_pipeline.py` — Xử lý mỗi window

### Stage 1: OCSVM Gate

```python
features = extract_ocsvm_features(samples, sample_rate)  # → 43 đặc trưng
ocsvm_label, ocsvm_score = self._ocsvm.predict_gate(features)
# ocsvm_label = "inlier" hoặc "outlier"
# ocsvm_score = số thực (> 0 → inlier, < 0 → outlier)
```

Nếu `outlier` và `bypass_ocsvm = False`:

```python
return AudioInferenceResult(
    ocsvm_label="outlier",
    gasoline_prob=0.0,      # trả về 0, không cho CNN chạy
    cnn_confidence=0.0,
    entropy=1.0,            # entropy cao = rất không chắc
    ...
)
```

Nếu `bypass_ocsvm = True` (debug mode): bỏ qua kiểm tra, CNN vẫn chạy.

### Stage 2: CNN Classification

```python
# Xây PCEN spectrogram từ waveform:
pcen = self._build_pcen(samples)
# Shape: (n_mels=128, target_frames=64) = 128×64 mel spectrogram

# Dự đoán:
probs = self._cnn.predict(pcen)
# probs = [noise_prob, gasoline_prob] = [0.381, 0.619]

noise_prob, gasoline_prob = probs[0], probs[1]
entropy = normalized_entropy(probs)
# entropy ≈ 0 → rất chắc chắn; entropy ≈ 1 → rất không chắc
```

**PCEN (Per-Channel Energy Normalization)** là kỹ thuật xử lý spectrogram giúp CNN hoạt động tốt hơn với âm thanh có nền động (như tiếng ồn cổng xe).

### `AudioInferenceResult` — kết quả 1 window

```python
@dataclass
class AudioInferenceResult:
    ocsvm_label: str        # "inlier" hoặc "outlier"
    gasoline_prob: float    # xác suất CNN cho là xăng [0, 1]
    noise_prob: float       # xác suất CNN cho là noise [0, 1]
    cnn_confidence: float   # max(probs) — confidence cao nhất
    entropy: float          # [0, 1] — 0 = chắc, 1 = không chắc
    window_start: float     # giây bắt đầu window
    window_end: float       # giây kết thúc window
```

---

## `inference/feature_extractor.py` — 43 đặc trưng OCSVM

```python
def extract_ocsvm_features(samples, sample_rate) -> np.ndarray:  # shape (43,)
```

43 đặc trưng được trích theo thứ tự:

| Nhóm            | Số đặc trưng | Ý nghĩa                                                                   |
| --------------- | ------------ | ------------------------------------------------------------------------- |
| **MFCC**        | 13           | Mel-Frequency Cepstral Coefficients — mô tả "màu âm" của giọng/tiếng động |
| **Delta-MFCC**  | 13           | Đạo hàm của MFCC theo thời gian — mô tả tốc độ thay đổi âm thanh          |
| **Chroma STFT** | 12           | Năng lượng trên 12 nốt nhạc — phân biệt âm điều hòa vs tạp âm             |
| **Spectral**    | 5            | Centroid, Bandwidth, Rolloff, ZCR, RMS — đặc trưng năng lượng phổ tần     |

```python
# Tất cả đều là MEAN theo trục thời gian (trung bình qua các frame)
mfcc_mean    = np.mean(librosa.feature.mfcc(...),          axis=1)  # (13,)
delta_mean   = np.mean(librosa.feature.delta(mfcc),        axis=1)  # (13,)
chroma_mean  = np.mean(librosa.feature.chroma_stft(...),   axis=1)  # (12,)
# + 5 spectral features = ZCR mean, RMS mean, centroid, bandwidth, rolloff

feature_vector = concat([mfcc_mean, delta_mean, chroma_mean, spectral])
# → shape (43,)
```

---

## `inference/ocsvm_model.py` — One-Class SVM

```python
class OCSVMModel:
    def predict_gate(self, feature_vector: np.ndarray) -> tuple[str, float]:
        score = float(self._model.decision_function(X)[0])
        label = "inlier" if score > 0.0 else "outlier"
        return label, score
```

### decision_function là gì?

One-Class SVM học **biên giới** bao quanh dữ liệu training (âm thanh micro thật).  
`decision_function` trả về **khoảng cách có dấu** đến biên giới đó:

```
  Training data (âm thanh motor xe từ mic thật):
                    ┌─────────────────────────────┐
  INLIER (score>0) │  ●  ●  ●  ●  ●  ●  ●  ●  ● │ INLIER
                    │     ●  ●  ●  ●  ●  ●  ●     │
                    └─────────────────────────────┘
  OUTLIER (score<0)     ×                    × OUTLIER
                    ×   (tiếng gió, mp4 codec, ×
                         tiếng người nói...)

  score > 0 → bên trong → inlier  → CNN tiếp tục
  score < 0 → bên ngoài → outlier → CNN bị chặn
```

---

## `inference/cnn_model.py` — CNN với SpatialAttention

### Kiến trúc

Model là CNN xử lý **PCEN mel spectrogram** (128 × 64 × 1 — ảnh 128 hàng x 64 cột 1 kênh), có thêm lớp `SpatialAttentionBlock` tự chế.

```python
class SpatialAttentionBlock(layers.Layer):
    """
    Học "vùng nào trên spectrogram quan trọng" → nhân vào feature map.
    Giúp CNN tập trung vào dải tần quan trọng của tiếng động cơ.
    """
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs,  axis=-1, keepdims=True)
        concat   = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)          # conv 7×7 → sigmoid → [0,1]
        return inputs * attention              # scale feature map theo attention
```

### Đầu ra binary sigmoid

Model output shape là `(1,)` — một số [0,1] là **xác suất gasoline**:

```python
def predict(self, pcen_spectrogram) -> np.ndarray:
    model_input = np.expand_dims(pcen_spectrogram, axis=(0, -1))
    # shape: (1, 128, 64, 1)

    raw = self._model.predict(model_input, verbose=0)
    # raw = [[0.619]]  → chỉ 1 số

    gas = float(raw.flatten()[0])   # 0.619 = prob gasoline
    return np.array([1.0 - gas, gas])  # → [0.381, 0.619] = [noise, gasoline]
```

---

## `aggregation/audio_aggregation.py` — Tổng hợp kết quả

Sau khi xử lý 5 windows độc lập → tổng hợp thành 1 kết quả cho cả event:

```python
def aggregate_audio_windows(results: list[AudioInferenceResult]) -> AggregatedAudio:
    # Lọc chỉ lấy inlier windows (skip outlier)
    inliers = [r for r in results if r.ocsvm_label == "inlier"]

    if not inliers:
        return AggregatedAudio(audio_gate=0.0, ...)  # không dùng audio

    # Tính trung bình có trọng số theo confidence
    gasoline_prob = mean([r.gasoline_prob  for r in inliers])
    entropy       = mean([r.entropy        for r in inliers])

    return AggregatedAudio(
        audio_gate=1.0,           # có ít nhất 1 inlier → audio hợp lệ
        gasoline_prob=0.62,
        entropy=0.59,
        num_windows=5,
        num_inlier_windows=5,
    )
```

**`audio_gate`** là switch nhị phân:

- `0.0` → tất cả windows đều outlier → không dùng audio trong fusion
- `1.0` → có ít nhất 1 inlier → dùng audio

---

## `fusion/fusion_engine.py` — Kết hợp Vision + Audio

### Công thức fusion

```python
# Inputs:
V_gas  = vision_summary["gasoline"]   # 0.798
V_elec = vision_summary["electric"]  # 0.202
A_gas  = audio_result.gasoline_prob  # 0.619
gate   = audio_result.audio_gate     # 1.0 (hoặc 0.0)
H      = audio_result.entropy        # 0.592

# Trọng số (từ config.yaml):
w_V = 0.6    # vision weight
w_A = 0.4    # audio weight

# Contribution của audio (bị giảm nếu entropy cao, bị tắt nếu gate=0):
audio_contrib = (1.0 - H) * gate
# = (1 - 0.592) × 1.0 = 0.408

# Fusion scores:
S_gasoline = w_V × V_gas  + w_A × A_gas        × audio_contrib
           = 0.6 × 0.798  + 0.4 × 0.619        × 0.408
           = 0.479        + 0.101
           = 0.580

S_electric = w_V × V_elec + w_A × (1-A_gas)   × audio_contrib
           = 0.6 × 0.202  + 0.4 × 0.381        × 0.408
           = 0.121        + 0.062
           = 0.183

# Quyết định:
gap = |0.580 - 0.183| = 0.397
if gap < delta (0.20):
    label = "uncertain"
elif S_gasoline > S_electric:
    label = "gasoline"   ← trường hợp này
```

### Trường hợp không có audio (gate=0)

```python
# audio_contrib = (1-H) × 0.0 = 0.0
S_gasoline = 0.6 × 0.798 + 0.4 × 0.619 × 0.0 = 0.479
S_electric = 0.6 × 0.202 + 0.4 × 0.381 × 0.0 = 0.121
# → vision-only decision, Audio bị bỏ qua hoàn toàn
```

### Khi entropy cao (audio không chắc chắn)

```
entropy = 0.9 (CNN gần 50/50 giữa gasoline và noise):
audio_contrib = (1 - 0.9) × 1.0 = 0.10   ← audio đóng góp rất ít
```

Entropy cao → audio không đáng tin → giảm weight audio trong fusion → vision chiếm ưu thế.

---

## `config/config.yaml` — Cấu hình audio module

```yaml
# Model paths (relative to project root)
ocsvm_model_path: "models/ocsvm_pipeline1.pkl"
cnn_model_path: "models/gasoline_classifier_v3.h5"

# Audio processing
sample_rate: 16000 # phải khớp với sample rate training data
fmax: 8000 # tần số tối đa (Nyquist = sr/2)
hop_length: 256 # step size của STFT
n_mels: 128 # số mel filter banks
target_frames: 64 # số time frames của PCEN spectrogram

# Sliding window
window_duration: 2.0 # mỗi window dài 2 giây
window_stride: 1.0 # bước trượt 1 giây (overlap 50%)

# CNN class mapping (phải khớp với thứ tự class khi training)
class_label_map:
  0: "noise"
  1: "gasoline"

# Fusion weights
w_vision: 0.6 # tỉ trọng vision (60%)
w_audio: 0.4 # tỉ trọng audio (40%)
delta: 0.20 # margin để kết luận "uncertain"

# Debug flag
bypass_ocsvm: true # true = mp4 test, false = production mic thật
```
