# Tổng quan dự án — Vehicle Entry Detection Pipeline

## 1. Mục tiêu

Hệ thống tự động nhận diện xe vào cổng từ **một camera duy nhất** quay vào cổng, xác định:

1. **Biển số xe** — bằng YOLO detect + OCR
2. **Loại nhiên liệu** — xăng hay điện — bằng kết hợp hình ảnh (vision) + âm thanh (audio)
3. **Gửi kết quả** lên backend server qua REST API

Đầu vào: file **video .mp4** (test) hoặc camera stream thật (production).  
Đầu ra: `{ plate, vehicle_type, timestamp, scores }` → POST lên server.

---

## 2. Cấu trúc folder

```
d:\audio_model\
│
├── audio_module/              ← Toàn bộ logic xử lý âm thanh (có thể dùng độc lập)
│   ├── __init__.py            — Export AudioEngine ra ngoài
│   ├── audio_engine.py        — Orchestrator: nhận waveform → trả fusion result
│   ├── aggregation/
│   │   └── audio_aggregation.py  — Tổng hợp kết quả nhiều window → 1 AggregatedAudio
│   ├── config/
│   │   ├── config.py          — DataClass AudioModuleConfig
│   │   └── config.yaml        — File cấu hình thực tế (model paths, thông số)
│   ├── fusion/
│   │   └── fusion_engine.py   — Kết hợp vision score + audio score → final label
│   └── inference/
│       ├── audio_pipeline.py  — Per-window: OCSVM gate → CNN classify
│       ├── cnn_model.py       — Load & run CNN (Keras + SpatialAttention)
│       ├── feature_extractor.py — Trích 43 đặc trưng âm thanh cho OCSVM
│       └── ocsvm_model.py     — Load & run One-Class SVM (scikit-learn/joblib)
│
├── vehicle_pipeline/          ← Pipeline chính chạy trên video
│   ├── main.py                — Entry point, main loop xử lý frame-by-frame
│   ├── config.py              — DataClass PipelineConfig (tất cả tham số pipeline)
│   ├── plate_pipeline.py      — YOLO detect biển số + OCR + voting
│   ├── vision_classifier.py   — Phân loại xăng/điện từ ảnh đuôi xe
│   ├── event_manager.py       — State machine: IDLE → CAPTURING → PROCESSING
│   ├── audio_extractor.py     — Trích audio từ mp4 bằng ffmpeg → numpy array
│   └── api_client.py          — POST kết quả lên server (với auto-retry)
│
├── helpers/                   ← Tiện ích OCR biển số
│   ├── ocr.py                 — Hàm read_plate(): nhận ảnh crop → trả (text, conf)
│   └── utils_rotate.py        — Deskew biển số bị nghiêng trước khi OCR
│
├── constant/
│   └── ocr.py                 — CLASS_NAMES: 31 ký tự biển số VN (phải đúng thứ tự training)
│
├── models/                    ← Model weights (không commit lên git)
│   ├── model_lp.pt            — YOLO detect vùng biển số
│   ├── model_ocr.pt           — YOLO OCR ký tự biển số
│   ├── detect_type_vehicle.pt — YOLO classify xăng/điện từ ảnh
│   ├── gasoline_classifier_v3.h5 — CNN phân loại âm thanh (xăng vs noise)
│   └── ocsvm_pipeline1.pkl    — One-Class SVM lọc âm thanh không hợp lệ
│
├── debug_audio/               ← File .wav debug (tự sinh khi keep_debug_audio=True)
├── temp_audio/                ← File .wav tạm thời (tự xóa sau mỗi event)
│
├── run.ps1                    ← Script PowerShell để chạy nhanh (set ENV + gọi python)
└── requirements.txt           ← Danh sách thư viện cần cài
```

---

## 3. Phụ thuộc chính

| Thư viện                  | Dùng cho                                             |
| ------------------------- | ---------------------------------------------------- |
| `ultralytics`             | YOLO detect biển số, OCR, classify xe                |
| `opencv-python`           | Đọc video frame-by-frame, vẽ bounding box            |
| `tensorflow` / `keras`    | Load và chạy CNN model (`gasoline_classifier_v3.h5`) |
| `scikit-learn` + `joblib` | Load và chạy OCSVM model (`.pkl`)                    |
| `librosa`                 | Trích đặc trưng âm thanh (MFCC, Chroma, ...)         |
| `ffmpeg` (system)         | Trích audio từ file mp4 → .wav                       |
| `requests`                | Gửi kết quả lên server qua HTTP POST                 |
| `pyyaml`                  | Đọc file `config.yaml`                               |

---

## 4. Cách chạy nhanh

```powershell
# Chạy với script (khuyên dùng)
.\run.ps1

# Truyền video khác
.\run.ps1 -Video "D:\videos\xe_test.mp4"

# Tắt bypass OCSVM (production với mic thật)
.\run.ps1 -BypassOcsvm false

# Gọi trực tiếp Python
python vehicle_pipeline/main.py --video "D:\zalo_cloud\xe6.mp4" --bypass-ocsvm true
```

`run.ps1` tự động set các biến môi trường cần thiết:

```powershell
$env:TF_CPP_MIN_LOG_LEVEL  = "3"   # tắt log verbose của TensorFlow
$env:TF_ENABLE_ONEDNN_OPTS = "0"   # tắt oneDNN warnings
```

---

## 5. Models cần có

Đặt tất cả vào folder `models/` (cùng cấp với `vehicle_pipeline/`):

| File                        | Kích cỡ ước tính | Mô tả                                  |
| --------------------------- | ---------------- | -------------------------------------- |
| `model_lp.pt`               | ~6 MB            | YOLO detect vùng biển số               |
| `model_ocr.pt`              | ~6 MB            | YOLO nhận dạng ký tự trên biển         |
| `detect_type_vehicle.pt`    | ~6 MB            | YOLO classify xăng/điện từ ảnh đuôi xe |
| `gasoline_classifier_v3.h5` | ~5 MB            | CNN binary sigmoid phân loại âm thanh  |
| `ocsvm_pipeline1.pkl`       | ~1 MB            | OCSVM lọc âm thanh nền/nhiễu           |
