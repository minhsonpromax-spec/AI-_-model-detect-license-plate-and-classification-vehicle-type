from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # ── Video source ─────────────────────────────────────────────────────────
    video_path: str = r"E:\video_ket_qua.mp4"

    # ── Frame processing ──────────────────────────────────────────────────────
    frame_skip: int = 2               # xử lý 1 frame trong mỗi N frame

    # ── Event lifecycle ───────────────────────────────────────────────────────
    event_idle_timeout: float = 4.0   # giây không thấy xe → kết thúc event
    event_cooldown: float = 5.0       # giây cooldown tối thiểu giữa 2 event cùng biển số

    # ── Plate detection thresholds ────────────────────────────────────────────
    plate_min_score: float = 1.5      # tổng điểm tối thiểu (normal: 2+ lần đọc)
    plate_min_ratio: float = 0.65     # tỉ lệ vote tối thiểu (biển số chiếm bao nhiêu % tổng)
    plate_fast_conf: float = 0.80     # fallback cho xe đi nhanh: 1 lần đọc conf >= 0.80
    plate_box_min_w: int = 50         # bỏ bounding box quá nhỏ (pixel)
    plate_box_min_h: int = 20
    yolo_conf: float = 0.5

    # ── Model paths ───────────────────────────────────────────────────────────
    yolo_lp_detect: str = "models/model_lp.pt"              # detect vùng biển số (YOLO track)
    yolo_lp_ocr: str = "models/model_ocr.pt"                # OCR ký tự biển số
    vision_model_path: str = "models/detect_type_vehicle.pt" # xăng/điện từ ảnh đuôi xe
    audio_config: str = "audio_module/config/config.yaml"                 # config của audio_model

    # ── Audio extraction ──────────────────────────────────────────────────────
    audio_temp_dir: str = "temp_audio"        # thư mục file .wav tạm (tự xóa)
    keep_debug_audio: bool = True             # True → lưu .wav vào debug_audio/ để nghe
    debug_audio_dir: str = "debug_audio"      # thư mục chứa .wav debug (t tự clear)

    # ── API ───────────────────────────────────────────────────────────────────
    api_url: str = "http://your-api.com/vehicle-events"
    api_timeout: float = 5.0
    api_retries: int = 2
