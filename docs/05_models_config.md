# Models & Configuration

## 1. Danh sách models

| File                        | Vị trí    | Framework             | Nhiệm vụ                           |
| --------------------------- | --------- | --------------------- | ---------------------------------- |
| `model_lp.pt`               | `models/` | YOLOv8 (ultralytics)  | Detect vùng biển số trong frame    |
| `model_ocr.pt`              | `models/` | YOLOv8 (ultralytics)  | Nhận dạng ký tự trên biển số       |
| `detect_type_vehicle.pt`    | `models/` | YOLOv8 classify       | Phân loại xe xăng / xe điện từ ảnh |
| `gasoline_classifier_v3.h5` | `models/` | Keras / TensorFlow    | CNN binary: xăng vs noise từ audio |
| `ocsvm_pipeline1.pkl`       | `models/` | scikit-learn (joblib) | OCSVM: lọc audio không hợp lệ      |

---

## 2. Chi tiết từng model

### `model_lp.pt` — YOLO detect biển số

- **Input:** Frame BGR (bất kỳ kích thước)
- **Output:** Bounding boxes `[x1, y1, x2, y2, conf, class]`
- **Dùng với:** `detect.track()` mode — gán track_id ổn định qua các frame
- **Threshold:** `conf >= 0.5` (cấu hình `yolo_conf`)

```python
results = self._detector.track(frame, persist=True, conf=0.5, verbose=False)[0]
for box in results.boxes:
    track_id = int(box.id)          # ID ổn định qua frame
    x1, y1, x2, y2 = box.xyxy[0]   # tọa độ bounding box
```

---

### `model_ocr.pt` — YOLO OCR ký tự biển số

- **Input:** Ảnh crop biển số (BGR, bất kỳ kích thước)
- **Output:** Ký tự biển số + confidence
- **Class mapping:** 31 ký tự biển số Việt Nam (xem `constant/ocr.py`)
- **Thứ tự class phải khớp với training** — nếu model re-train cần kiểm tra lại

```python
# Kiểm tra thứ tự class của model:
python -c "from ultralytics import YOLO; m = YOLO('models/model_ocr.pt'); print(m.names)"

# Kết quả mẫu:
# {0: '0', 1: '1', ..., 9: '9', 10: 'A', ..., 35: 'Z', ...}
```

**31 ký tự VN biển số** (`constant/ocr.py`):

```
0 1 2 3 4 5 6 7 8 9
A B C D E F G H K L
M N P R S T U V X Y Z
Đ
```

_(Không có I, O, Q, W vì dễ nhầm lẫn với 0, 1)_

**Deskew trước khi OCR:**  
Biển số đôi khi bị nghiêng trong ảnh crop → `helpers/utils_rotate.py` thử deskew 4 hướng:

```python
for cc in range(2):      # correction config 0 hoặc 1
    for ct in range(2):  # correction type 0 hoặc 1
        rotated = utils_rotate.deskew(crop, cc, ct)
        lp, conf = ocr.read_plate(self._ocr_model, rotated)
        if lp != "unknown":
            return lp, conf   # lấy kết quả đầu tiên hợp lệ
```

---

### `detect_type_vehicle.pt` — YOLO classify xăng/điện

- **Input:** Frame BGR đuôi xe
- **Output:** Xác suất từng class `[p_gasoline, p_electric]`
- **Mode:** YOLO classification (không phải detection)
- **Giả định:** class 0 = gasoline, class 1 = electric

```python
results = self._model(frame, verbose=False)[0]
probs = results.probs.data.cpu().numpy()
# probs = [0.798, 0.202]
return {"gasoline": probs[0], "electric": probs[1]}
```

> **Lưu ý:** Nếu kết quả ngược chiều (xe điện ra xăng cao), đổi index:
>
> ```python
> return {"gasoline": probs[1], "electric": probs[0]}
> ```

---

### `gasoline_classifier_v3.h5` — CNN phân loại âm thanh

- **Input:** PCEN mel spectrogram shape `(1, 128, 64, 1)`  
  _(1 batch × 128 mel bins × 64 time frames × 1 channel)_
- **Output:** Scalar `[[0.619]]` — xác suất là âm thanh xe xăng
- **Architecture:** CNN + `SpatialAttentionBlock` (custom Keras layer)
- **Output type:** Binary sigmoid (không phải softmax 2 class)

```
Input (1, 128, 64, 1)
  │
  ├── Conv2D blocks
  │── SpatialAttentionBlock  ← học "dải tần nào quan trọng"
  │── GlobalAveragePooling
  └── Dense(1, sigmoid)
        │
        ▼
    [[0.619]]   ← xác suất gasoline
```

Conversion sang 2-class trong code:

```python
gas = float(raw.flatten()[0])   # 0.619
return np.array([1.0 - gas, gas])  # → [0.381, 0.619]
#                  ↑               ↑
#               noise_prob    gasoline_prob
```

---

### `ocsvm_pipeline1.pkl` — OCSVM filter

- **Input:** Feature vector shape `(43,)` — trích từ waveform 2s
- **Output:** `(label, score)` — `("inlier", 0.3)` hoặc `("outlier", -8.5)`
- **Framework:** scikit-learn `OneClassSVM`, lưu bằng `joblib`
- **Training data:** Âm thanh động cơ xe thu bằng **microphone thật** ở cổng thực tế

```python
model = joblib.load("models/ocsvm_pipeline1.pkl")
score = float(model.decision_function(features.reshape(1, -1))[0])
label = "inlier" if score > 0.0 else "outlier"
```

**Score distribution thực tế:**

| Nguồn âm thanh    | Score OCSVM   | Phân loại  |
| ----------------- | ------------- | ---------- |
| Mic thật, xe xăng | +0.2 đến +0.8 | ✅ Inlier  |
| Mic thật, xe điện | -0.1 đến +0.3 | ✅ Inlier  |
| File mp4 (test)   | -7.0 đến -8.5 | ❌ Outlier |
| Tiếng ồn nền      | -2.0 đến -5.0 | ❌ Outlier |

---

## 3. Cấu hình pipeline (`vehicle_pipeline/config.py`)

Tất cả tham số pipeline nằm trong `PipelineConfig` với giá trị mặc định:

### Nhóm Video

| Tham số      | Default                 | Giải thích                            |
| ------------ | ----------------------- | ------------------------------------- |
| `video_path` | `D:\zalo_cloud\xe6.mp4` | Đường dẫn file mp4                    |
| `frame_skip` | `2`                     | Xử lý 1 trong mỗi 2 frame → tốc độ x2 |

### Nhóm Event lifecycle

| Tham số              | Default | Giải thích                                   |
| -------------------- | ------- | -------------------------------------------- |
| `event_idle_timeout` | `4.0s`  | Không thấy xe trong bao lâu → kết thúc event |
| `event_cooldown`     | `5.0s`  | Chờ tối thiểu trước khi gửi lại cùng biển số |

### Nhóm Plate detection

| Tham số           | Default | Giải thích                                   |
| ----------------- | ------- | -------------------------------------------- |
| `plate_min_score` | `1.5`   | Tổng weight-vote tối thiểu để accept biển số |
| `plate_min_ratio` | `0.65`  | Biển số phải chiếm ≥65% tổng điểm vote       |
| `plate_fast_conf` | `0.80`  | FastCar: 1 lần đọc nhưng conf ≥ 0.80         |
| `plate_box_min_w` | `50px`  | Bỏ box quá nhỏ (nhiễu xa)                    |
| `yolo_conf`       | `0.5`   | Confidence tối thiểu YOLO detect             |

**Tuning `event_idle_timeout`:**

- Quá ngắn (< 2s): xe chậm bị cắt event giữa chừng
- Quá dài (> 6s): 2 xe liên tiếp bị merge thành 1 event
- `4.0s` là điểm cân bằng cho cổng bình thường

**Tuning `plate_min_score`:**

- Quá cao: xe nhanh bị từ chối (không đủ lần đọc)
- Quá thấp: nhiễu có thể được accept nhầm
- `1.5` ≈ 2 lần đọc với conf ~0.75

---

## 4. Cấu hình audio (`audio_module/config/config.yaml`)

### Nhóm Audio processing

| Tham số           | Default | Giải thích                             |
| ----------------- | ------- | -------------------------------------- |
| `sample_rate`     | `16000` | Hz — phải khớp training                |
| `n_mels`          | `128`   | Số mel filter banks cho spectrogram    |
| `target_frames`   | `64`    | Số time frames output PCEN             |
| `hop_length`      | `256`   | Step size STFT = 256/16000 ≈ 16ms      |
| `window_duration` | `2.0s`  | Độ dài mỗi sliding window              |
| `window_stride`   | `1.0s`  | Bước trượt giữa 2 window (50% overlap) |

### Nhóm Fusion weights

| Tham số    | Default | Giải thích                                 |
| ---------- | ------- | ------------------------------------------ |
| `w_vision` | `0.6`   | Vision đóng góp 60% vào decision           |
| `w_audio`  | `0.4`   | Audio đóng góp 40% (khi gate=1, entropy=0) |
| `delta`    | `0.20`  | Gap tối thiểu để confirm (tránh uncertain) |

**Ví dụ điều chỉnh:**  
Nếu muốn tin tưởng audio hơn vision (khi OCSVM đã được retrain tốt):

```yaml
w_vision: 0.4
w_audio: 0.6
```

### Nhóm Debug

| Tham số        | Default | Giải thích                                |
| -------------- | ------- | ----------------------------------------- |
| `bypass_ocsvm` | `true`  | Bỏ qua OCSVM gate (dùng khi test với mp4) |

> **Production checklist:**  
> Trước khi deploy với mic thật:
>
> - [ ] Retrain OCSVM với audio từ thiết bị thực tế
> - [ ] Đặt `bypass_ocsvm: false` trong `config.yaml`
> - [ ] Hoặc chạy `.\run.ps1 -BypassOcsvm false`

---

## 5. Biến môi trường

Set trong `run.ps1` hoặc thủ công trước khi chạy:

| Biến                    | Giá trị | Tác dụng                                           |
| ----------------------- | ------- | -------------------------------------------------- |
| `TF_CPP_MIN_LOG_LEVEL`  | `3`     | Tắt TensorFlow verbose log (chỉ giữ ERROR)         |
| `TF_ENABLE_ONEDNN_OPTS` | `0`     | Tắt oneDNN optimization warning                    |
| `NUMBA_DISABLE_JIT`     | `1`     | Tắt JIT compile của numba → librosa load nhanh hơn |

```powershell
# Dùng run.ps1 (khuyên dùng):
.\run.ps1

# Hoặc set thủ công:
$env:TF_CPP_MIN_LOG_LEVEL  = "3"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:NUMBA_DISABLE_JIT     = "1"
python vehicle_pipeline/main.py --video "D:\videos\test.mp4" --bypass-ocsvm true
```
