# Chi tiết Vehicle Pipeline — `vehicle_pipeline/`

Folder này chứa toàn bộ logic **điều phối pipeline** từ video → kết quả.  
Mỗi file phụ trách một nhiệm vụ riêng biệt, ghép lại trong `main.py`.

---

## `config.py` — Tất cả tham số trong một chỗ

```python
@dataclass
class PipelineConfig:
    video_path: str = r"D:\zalo_cloud\xe6.mp4"

    frame_skip: int = 2               # xử lý 1/2 số frame (giảm CPU load)
    event_idle_timeout: float = 4.0   # giây không thấy xe → kết thúc event
    event_cooldown: float = 5.0       # chặn gửi duplicate cùng biển số

    plate_min_score: float = 1.5      # threshold voting biển số (normal)
    plate_min_ratio: float = 0.65     # tỉ lệ vote tối thiểu
    plate_fast_conf: float = 0.80     # threshold FastCar (1 lần đọc conf cao)
    plate_box_min_w: int = 50         # lọc bounding box quá nhỏ
    plate_box_min_h: int = 20
    yolo_conf: float = 0.5            # ngưỡng confidence YOLO detect

    yolo_lp_detect: str = "models/model_lp.pt"
    yolo_lp_ocr: str = "models/model_ocr.pt"
    vision_model_path: str = "models/detect_type_vehicle.pt"
    audio_config: str = "audio_module/config/config.yaml"

    keep_debug_audio: bool = True     # True → lưu .wav vào debug_audio/
    debug_audio_dir: str = "debug_audio"
    audio_temp_dir: str = "temp_audio"

    api_url: str = "http://your-api.com/vehicle-events"
    api_timeout: float = 5.0
    api_retries: int = 2
```

**Cách dùng:** Mọi tham số đều có giá trị mặc định → chạy được luôn mà không cần config.  
Muốn thay đổi thì truyền vào khi khởi tạo: `PipelineConfig(video_path="...", frame_skip=3)`.

---

## `plate_pipeline.py` — Detect biển số + OCR + Voting

### Luồng trong 1 frame

```
frame (BGR numpy)
       │
       ▼
YOLO.track(frame, persist=True)   ← model_lp.pt
       │
       │  trả về list boxes, mỗi box có:
       │  - xyxy: tọa độ (x1, y1, x2, y2)
       │  - id:   track_id (ổn định qua các frame nhờ persist=True)
       │  - conf: confidence detect
       │
       ▼ (mỗi box)
crop = frame[y1:y2, x1:x2]         ← cắt vùng biển số
       │
       ▼
_read_plate(crop)
  → thử deskew 4 cách (cc=0/1, ct=0/1)
  → YOLO(rotated_crop, verbose=False)   ← model_ocr.pt
  → lấy kết quả đọc được đầu tiên (không phải "unknown")
  → trả về (text, confidence)
       │
       ▼
cache[track_id].append((text, conf))   ← lưu lịch sử theo track
recent_queue.append((text, conf))      ← rolling window mới nhất (15 reads)
```

### Voting — `_vote(plates)`

```python
# Ví dụ cache tích lũy trong 1 event:
plates = [
    ("75AA-37010", 0.86),
    ("75AA-37010", 0.83),
    ("75AA-37012", 0.79),   # OCR nhầm số cuối
    ("75AA-37010", 0.91),
]

# Weighted vote:
score_map = {
    "75AA-37010": 0.86 + 0.83 + 0.91 = 2.60,
    "75AA-37012": 0.79,
}
total_score = 3.39
best = "75AA-37010"
ratio = 2.60 / 3.39 = 0.767

# Cơ chế ① Normal: score >= 1.5 AND ratio >= 0.65 → PASS ✓

# Cơ chế ② FastCar (xe nhanh chỉ đọc 1-2 lần):
# best_conf = 0.91 >= 0.80  AND  ratio = 100% >= 90% → PASS ✓
```

### Phát hiện xe mới — `get_dominant_prefix()`

```python
# recent_queue 15 phần tử gần nhất:
# ["75AA-37010", "75AA-37012", "75AA-37010", "74G1-20202", "74G1-20200", "74G1-20202"]
#                                             ↑── xe mới bắt đầu xuất hiện

# Lấy prefix (phần trước dấu "-"):
# "75AA-37010" → "75AA"
# "74G1-20202" → "74G1"

# Kiểm tra 3 lần đọc CUỐI CÙNG:
tail = ["74G1-20202", "74G1-20200", "74G1-20202"]
prefixes = ["74G1", "74G1", "74G1"]
# Tất cả giống nhau → dominant_prefix = "74G1"

# committed_prefix của event hiện tại là "75AA"
# "74G1" ≠ "75AA" → detect_new_vehicle() = True → force_end_capture()
```

**Tại sao dùng prefix thay vì full text?**  
Phần số cuối (`37010`) hay bị OCR đọc nhầm (`37012`, `3701O`).  
Phần tỉnh/serie đầu (`75AA`, `74G1`) ổn định hơn nhiều.

---

## `event_manager.py` — State Machine

```
                 on_detection()
    ┌────────────────────────────────────────────┐
    │                                            │
  IDLE ─────────────────────────────────► CAPTURING
                                               │  │
                             should_end_capture()  │ force_end_capture()
                             (timeout)             │ (xe mới phát hiện)
                                               ▼  ▼
                                          PROCESSING
                                               │
                                    finish_processing()
                                               │
                                             IDLE
```

### Các phương thức quan trọng

```python
# Gọi mỗi frame có detect xe
event_manager.on_detection(current_sec)
# → nếu IDLE: tạo VehicleEvent mới (start_sec = now), chuyển sang CAPTURING
# → nếu CAPTURING: cập nhật last_detection_sec (để tính timeout)

# Gọi sau khi voting biển số lần đầu thành công
event_manager.commit_plate("75AA-37010", prefix="75AA")
# → lưu biển đại diện cho event này, dùng để so sánh xe mới

# Gọi cuối mỗi frame
event_manager.should_end_capture(current_sec)
# → True nếu (current_sec - last_detection_sec) > 4.0s

# Gọi khi detect xe mới (force end)
event_manager.force_end_capture(current_sec)
# → CAPTURING → PROCESSING ngay lập tức

# Gọi sau khi xử lý xong event
event_manager.finish_processing()
# → PROCESSING → IDLE, reset tất cả
```

### Cooldown per-plate

Bảo vệ tránh gửi duplicate khi camera liên tục thấy cùng một xe (vd: xe đậu lâu):

```python
event_manager.is_plate_in_cooldown("75AA-37010")
# → True nếu plate này đã được gửi trong vòng 5.0s qua
# → Nếu True: bỏ qua event, không gửi server

event_manager.mark_plate_sent("75AA-37010")
# → ghi nhận thời điểm gửi (dùng wall-clock time, không phải video time)
```

---

## `audio_extractor.py` — Trích audio từ mp4

```python
def extract_audio_segment(
    video_path: str,      # "D:/videos/xe6.mp4"
    start_sec: float,     # 4.53
    end_sec: float,       # 9.40
    sample_rate: int = 16000,
    keep_wav: bool = False,
) -> np.ndarray:          # float32 array, shape (77867,) = 4.87s × 16000
```

**Cách hoạt động:**

```
ffmpeg -y -ss 4.530000 -i xe6.mp4 -t 4.870000 -ar 16000 -ac 1 -f wav temp.wav
                ↑ seek trước -i để nhanh (keyframe seek)                    ↑ mono
       │
       ▼
librosa.load("temp.wav", sr=16000)
       │
       ▼
numpy float32 array, normalize về [-1.0, 1.0]
       │
       ├── keep_wav=True  → copy sang debug_audio/seg_4.53s_to_9.40s.wav
       └── keep_wav=False → xóa file tạm
```

**Lưu ý bảo mật:** Dùng `subprocess` với list arguments (không phải `shell=True`) → tránh command injection.

```python
cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path, ...]
subprocess.run(cmd, capture_output=True, check=True)
# Không dùng: subprocess.run(f"ffmpeg ... {video_path}", shell=True)  ← XẤU
```

---

## `vision_classifier.py` — Phân loại xăng/điện từ ảnh

```python
class RealVisionClassifier:
    def __init__(self, model_path: str):
        self._model = YOLO(model_path)   # detect_type_vehicle.pt

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        results = self._model(frame, verbose=False)[0]
        probs = results.probs.data.cpu().numpy()
        # probs = [0.798, 0.202]
        # index 0 = gasoline, index 1 = electric (theo thứ tự training)
        return {
            "gasoline": float(probs[0]),   # 0.798
            "electric": float(probs[1]),   # 0.202
        }
```

Model dùng là YOLO classification mode (không phải detection) → nhận cả frame → trả xác suất từng class.

---

## `api_client.py` — Gửi kết quả lên server

```python
# Payload gửi đi:
{
    "plate": "75AA-37010",
    "vehicle_type": "gasoline",
    "timestamp": 1744123456.789,    # UNIX timestamp
    "score_gasoline": 0.580,
    "score_electric": 0.183,
    "audio_gate": 1.0,
    "entropy": 0.592,
    "final_label": "gasoline"
}
```

**Auto-retry:** Tự động thử lại khi server trả `500/502/503/504`:

```python
Retry(
    total=2,              # tự động thử tối đa 2 lần
    backoff_factor=0.5,   # chờ 0.5s, 1.0s giữa các lần thử
    status_forcelist=[500, 502, 503, 504],  # chỉ retry lỗi server
    allowed_methods=["POST"],
)
```

Không retry lỗi `4xx` (404, 422...) vì đó là lỗi client, retry cũng vô ích.

---

## `main.py` — Entry point & Main loop

### Khởi động

```python
def run(config: PipelineConfig) -> None:
    # 1. Load models
    plate_pipeline  = PlatePipeline(...)    # YOLO detect + OCR
    vision_clf      = RealVisionClassifier(...) # YOLO classify
    audio_engine    = AudioEngine.from_yaml(...)
    api_client      = ApiClient(...)
    event_manager   = EventManager(config)

    # 2. Mở video
    cap = cv2.VideoCapture(config.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)         # 30.0
```

### Main loop (mỗi frame)

```python
while True:
    ret, frame = cap.read()
    if not ret:
        # hết video → xử lý event cuối nếu còn đang CAPTURING
        break

    frame_count += 1
    current_sec = frame_count / fps

    # Bỏ qua frame lẻ (frame_skip=2 → xử lý 1/2 frame)
    if frame_count % 2 != 0:
        continue

    # Đang xử lý event → bỏ qua detect tạm thời
    if event_manager.state == PROCESSING:
        continue

    # === Detect biển số ===
    has_vehicle = plate_pipeline.process_frame(frame, frame_count)

    if has_vehicle:
        event_manager.on_detection(current_sec)

        if event_manager.state == CAPTURING:
            # Thu thập vision score
            vision_score = vision_clf.predict(frame)
            event.vision_scores.append(vision_score)

            # Commit plate đại diện (lần đầu vote được)
            if not event_manager._committed_plate:
                rt_plate = plate_pipeline.get_best_plate_all()
                if rt_plate:
                    event_manager.commit_plate(rt_plate, prefix=...)

            # Kiểm tra xe mới (prefix thay đổi)
            dominant = plate_pipeline.get_dominant_prefix(min_streak=3)
            if dominant and event_manager.detect_new_vehicle(dominant):
                event_manager.force_end_capture(current_sec)

    # Kiểm tra timeout
    if event_manager.should_end_capture(current_sec):
        event_manager.end_capture(current_sec)

    # Xử lý nếu vừa kết thúc
    if event_manager.state == PROCESSING:
        _process_event(...)
        event_manager.finish_processing()
```

### `_process_event()` — Xử lý mỗi event

```python
def _process_event(event, config, ...):
    # 1. Lấy biển số tốt nhất từ cache
    plate = plate_pipeline.get_best_plate_all()

    # 2. Kiểm tra cooldown (tránh duplicate)
    if event_manager.is_plate_in_cooldown(plate):
        return

    # 3. Tính trung bình vision scores
    vision_summary = { "gasoline": mean(...), "electric": mean(...) }

    # 4. Extract audio + chạy AudioEngine
    waveform = extract_audio_segment(video, start_sec, end_sec)
    fusion_result = audio_engine.process(waveform, 16000, vision_summary)

    # 5. In kết quả đẹp
    # ┌─────────────────┐
    # │ BIỂN SỐ: ...    │
    # │ LOẠI XE: ...    │
    # └─────────────────┘

    # 6. POST lên server
    api_client.send_vehicle_event(plate, vehicle_type, extra=fusion_result)

    # 7. Ghi stats (để in bảng tổng kết cuối video)
    stats_list.append(EventStats(...))
```

### Bảng thống kê cuối video

```
══════════════════════════════════════════════════════════════════════
  THỐNG KÊ KẾT QUẢ PIPELINE  (bypass_ocsvm=ON)
══════════════════════════════════════════════════════════════════════
  Event #1
  ─────────────────────────────────────────────────────────────────────
  Biển số     : 75AA-37010
  Thời lượng  : 4.87 s

  [Vision]
    Frames      : 5
    Xăng        : 0.798   |  Điện: 0.202

  [Audio - OCSVM]
    Inlier      : 5 cửa sổ
    Outlier     : 0 cửa sổ

  [Audio - CNN]  (bypass=ON: chạy cả outlier window)
    Gas (all)   : 0.619  ← dùng khi bypass
    Gas (inlier): 0.619  ← dùng khi không bypass
    Audio gate  : 1.0   |  Entropy: 0.592

  [Fusion]
    Score xăng  : 0.580   |  Score điện: 0.183
    Kết quả     : GASOLINE
══════════════════════════════════════════════════════════════════════
  Tổng: 2 event  |  Xăng: 2  |  Điện: 0  |  Không rõ: 0
══════════════════════════════════════════════════════════════════════
```
