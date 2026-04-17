# Pipeline tổng thể — Luồng xử lý từ đầu đến cuối

## 1. Nhìn toàn cảnh

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VIDEO FILE (.mp4)                            │
│                   (camera quay vào cổng ra vào)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │  frame-by-frame (OpenCV)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MAIN LOOP  (main.py)                           │
│                                                                     │
│  ┌──────────────────┐   ┌────────────────┐   ┌───────────────────┐ │
│  │  PlatePipeline   │   │ VisionClassifier│   │  EventManager     │ │
│  │  YOLO detect     │──►│  YOLO classify  │──►│  State machine    │ │
│  │  + OCR + vote    │   │  xăng / điện    │   │  IDLE/CAPTURING/  │ │
│  └──────────────────┘   └────────────────┘   │  PROCESSING       │ │
│                                               └────────┬──────────┘ │
└────────────────────────────────────────────────────────│────────────┘
                                                         │ event end
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PROCESS EVENT  (main.py)                        │
│                                                                     │
│  Plate voting  →  Vision mean  →  Extract audio  →  AudioEngine    │
│                                        ▲                            │
│                                   ffmpeg + librosa                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO MODULE  (audio_module/)                    │
│                                                                     │
│  Sliding windows (2s)  →  OCSVM gate  →  CNN classify              │
│                                     ↓                               │
│                         FusionEngine: vision + audio → final label  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  API Client │  POST → Backend server
                    └─────────────┘
```

---

## 2. Khái niệm **Event**

Toàn bộ pipeline xoay quanh khái niệm **event** — một lần xe đi vào/qua cổng.

```
Timeline video:
  0s──────3s──────────────────9s──────────12s──────────────17s──────────
            [xe 1 vào]                      [xe 2 vào]
            ├── CAPTURING ──►|              ├── CAPTURING ──►|
            │  ghi nhận:     │              │  ghi nhận:     │
            │  - plate reads │  timeout 4s  │  - plate reads │
            │  - vision score│─────────────►│  - vision score│
            │  - audio range │   PROCESSING │  - audio range │
            └────────────────┘   → POST server└──────────────┘
```

Một event = một khoảng thời gian từ lúc **camera thấy biển số** cho đến khi **không thấy xe trong 4 giây**.

---

## 3. Các bước xử lý chi tiết

### Bước 1 — Đọc frame & detect biển số

```
Video frame (30fps)
       │
       │ frame_skip = 2 → chỉ xử lý frame chẵn (giảm 50% tài nguyên)
       ▼
PlatePipeline.process_frame(frame, frame_count)
       │
       ├── YOLO.track()  → detect vùng biển số + assign track_id
       │   model_lp.pt
       │
       └── YOLO(crop)    → OCR từng ký tự  (chạy mỗi 2 frame)
           model_ocr.pt
           │
           ▼
       cache[track_id].append( (text, confidence) )
       recent_queue.append( (text, confidence) )   ← rolling 15 reads
```

**Track ID** là ID mà YOLO gán cho bounding box qua các frame. Cùng 1 xe → cùng track_id.

---

### Bước 2 — State machine Event

```python
# Mỗi frame:
if has_vehicle:
    event_manager.on_detection(current_sec)   # IDLE → CAPTURING

# Kiểm tra kết thúc event:
if event_manager.should_end_capture(current_sec):
    event_manager.end_capture(current_sec)    # CAPTURING → PROCESSING
```

Ba trạng thái:

| State        | Ý nghĩa             | Hành động                                    |
| ------------ | ------------------- | -------------------------------------------- |
| `IDLE`       | Không có xe         | Chờ, không ghi nhận gì                       |
| `CAPTURING`  | Xe đang trong frame | Ghi plate, vision scores, theo dõi thời gian |
| `PROCESSING` | Xe đã rời frame     | Xử lý kết quả, tạm dừng detect xe mới        |

Điều kiện kết thúc CAPTURING (2 cơ chế):

```
Cơ chế 1 - Timeout:
  now - last_seen_sec > event_idle_timeout (4.0s)
  → xe đã rời frame, không thấy biển số trong 4 giây liên tục

Cơ chế 2 - Phát hiện xe mới (chống merge 2 xe):
  Prefix 3 lần OCR cuối = "74G1" ≠ committed_prefix "75AA"
  → xe khác đã vào, force end event cũ ngay lập tức
```

---

### Bước 3 — Voting biển số

Không dùng kết quả OCR từng frame (nhiễu cao), mà **tổng hợp nhiều lần đọc**:

```
Kết quả OCR trong suốt event:
  [("75AA-37010", 0.86), ("75AA-37010", 0.83), ("75AA-37012", 0.79), ...]
                                                           ↑
                                                    OCR noise trên số

Weighted voting:
  score("75AA-37010") = 0.86 + 0.83 = 1.69
  score("75AA-37012") = 0.79

Cơ chế pass:
  ① Normal  : score >= 1.5  AND  ratio >= 65%  → accepted
  ② FastCar : chỉ 1 lần đọc nhưng conf >= 0.80  AND  ratio >= 90%
              (xe đi nhanh qua cổng, chỉ kịp đọc 1-2 lần)
```

---

### Bước 4 — Vision classification

Trong lúc CAPTURING, mỗi frame **detect được xe** → chạy vision model:

```python
vision_score = vision_clf.predict(frame)
# → {"gasoline": 0.798, "electric": 0.202}

event.vision_scores.append(vision_score)
```

Cuối event → lấy **trung bình** tất cả frames:

```python
vision_summary = {
    "gasoline": mean([0.78, 0.82, 0.79, ...]),   # → 0.798
    "electric": mean([0.22, 0.18, 0.21, ...]),   # → 0.202
}
```

---

### Bước 5 — Extract audio

```python
waveform = extract_audio_segment(
    video_path = "xe6.mp4",
    start_sec  = event.start_sec,   # 4.53s
    end_sec    = event.end_sec,     # 9.40s
    sample_rate = 16000,
)
# Gọi ffmpeg subprocess → cắt đoạn 4.87s → đọc bằng librosa → numpy float32
```

Nếu `keep_debug_audio = True` → lưu `.wav` vào `debug_audio/seg_4.53s_to_9.40s.wav` để nghe kiểm tra.

---

### Bước 6 — Audio Engine (chi tiết ở docs/04)

```
waveform (4.87s, 16kHz)
       │
       ▼
Sliding windows: 5 windows × 2s (stride 1s)
  [-0.33s → 0.53s], [-0.33s → 1.53s], [0.53s → 2.53s], ...
       │
       ▼ (mỗi window)
OCSVM gate → CNN → gasoline_prob, entropy
       │
       ▼
Aggregation → AggregatedAudio
  audio_gate=1.0, gasoline_prob=0.62, entropy=0.59
       │
       ▼
FusionEngine
  S_gas = 0.6×0.798 + 0.4×0.62×(1-0.59)×1.0 = 0.580
  S_elec = 0.6×0.202 + 0.4×0.38×(1-0.59)×1.0 = 0.183
  gap = 0.397 > delta (0.20) → label = "gasoline"
```

---

### Bước 7 — Kết quả & API

```
┌─────────────────────────────────────────┐
│  BIỂN SỐ  : 75AA-37010                 │
│  LOẠI XE  : XE XĂNG 🔥                 │
└─────────────────────────────────────────┘

POST http://your-api.com/vehicle-events
{
  "plate": "75AA-37010",
  "vehicle_type": "gasoline",
  "timestamp": 1744123456.789,
  "score_gasoline": 0.580,
  "score_electric": 0.183,
  "audio_gate": 1.0,
  "entropy": 0.592
}
```

---

## 4. Xử lý edge cases

### 4.1 Xe đi nhanh (FastCar)

Xe không dừng lại, chỉ lướt qua cổng trong 1-2 giây → OCR chỉ đọc được 1-2 lần.  
Cơ chế `FastCar`: nếu 1 lần đọc có `conf >= 0.80` và chiếm `>= 90%` tổng đọc → accepted.

### 4.2 Hai xe vào liên tiếp

```
Xe 1: "75AA" committed vào EventManager
Xe 2 xuất hiện: recent_queue = [..., "74G1", "74G1", "74G1"]
Prefix "74G1" ≠ committed "75AA" → force_end_capture(event xe 1)
→ Process event xe 1 → Bắt đầu event xe 2
```

### 4.3 Biển số đọc noisy

OCR đôi khi đọc sai phần số cuối (`37010` → `37012`, `3701O`).  
Dùng **prefix** (`75AA`) thay vì full text để detect xe mới, vì prefix ổn định hơn phần số.

### 4.4 Hết video

```python
if not ret:   # cap.read() trả về False
    if event_manager.state == CAPTURING:
        event_manager.end_capture(current_sec)
        _process_event(...)   # xử lý event cuối trước khi thoát
    break
```

### 4.5 Cooldown per-plate

Bảo vệ không gửi duplicate khi camera thấy lại cùng biển số trong `event_cooldown = 5.0` giây.
