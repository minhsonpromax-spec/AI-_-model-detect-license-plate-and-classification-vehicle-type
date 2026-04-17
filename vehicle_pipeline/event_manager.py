"""
Event state machine cho 1 xe đi vào cổng.

State transitions:
    IDLE  ──[xe detect]──►  CAPTURING  ──[hết timeout]──►  PROCESSING  ──[xong]──►  IDLE

VehicleEvent lưu toàn bộ dữ liệu thu thập được trong lúc CAPTURING.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class EventState(Enum):
    IDLE       = auto()   # Không có xe, đang chờ
    CAPTURING  = auto()   # Đang ghi nhận (OCR, vision, âm thanh)
    PROCESSING = auto()   # Đang xử lý kết quả (không đọc frame mới)


@dataclass
class VehicleEvent:
    """Dữ liệu thu thập trong 1 event (1 xe vào cổng)."""

    start_sec: float                                       # giây bắt đầu trong video
    end_sec: float = 0.0                                   # giây kết thúc trong video

    # List of {"gasoline": float, "electric": float} — 1 entry mỗi frame vision
    vision_scores: list[dict[str, float]] = field(default_factory=list)


class EventManager:
    """
    Quản lý vòng đời event và cooldown per-plate.

    Cách dùng trong main loop:
        1. Mỗi frame có detect biển số  → on_detection(current_sec)
        2. Cuối mỗi frame               → check should_end_capture(current_sec)
        3. Nếu True                     → end_capture(current_sec)
        4. Khi state == PROCESSING      → xử lý event.current_event
        5. Sau khi xử lý xong          → finish_processing()
    """

    def __init__(self, config) -> None:
        self._cfg = config
        self._state: EventState = EventState.IDLE
        self._event: Optional[VehicleEvent] = None
        self._last_detection_sec: float = 0.0
        # Biển số đầu tiên được commit cho event đang capture (dùng để phát hiện xe mới)
        self._committed_plate: Optional[str] = None
        self._committed_prefix: Optional[str] = None  # prefix (ưu tiên hơn) để so sánh
        # plate_text → wall-clock time khi lần cuối gửi server
        self._plate_cooldown: dict[str, float] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> EventState:
        return self._state

    @property
    def current_event(self) -> Optional[VehicleEvent]:
        return self._event

    # ── State transitions ─────────────────────────────────────────────────────

    def on_detection(self, current_sec: float) -> None:
        """Gọi khi frame này có biển số được detect."""
        if self._state == EventState.IDLE:
            self._event = VehicleEvent(start_sec=current_sec)
            self._state = EventState.CAPTURING

        # Cập nhật lần nhìn thấy cuối cùng (dùng để tính timeout)
        self._last_detection_sec = current_sec

    def commit_plate(self, plate: str, prefix: Optional[str] = None) -> None:
        """Ghi nhận biển số đại diện cho event đang capture (chỉ set lần đầu)."""
        if self._committed_plate is None:
            self._committed_plate = plate
            self._committed_prefix = prefix or plate

    def detect_new_vehicle(self, dominant_prefix: str) -> bool:
        """
        True nếu dominant_prefix khác prefix đã commit → xe mới đi vào.

        Dùng prefix (phần tỉnh/serie trước '-') thây vì full text
        để chống lại OCR noise trên phần số cuối biển.
        """
        if self._state != EventState.CAPTURING:
            return False
        if self._committed_prefix is None:
            return False
        return dominant_prefix != self._committed_prefix

    def force_end_capture(self, current_sec: float) -> None:
        """Kết thúc event ngay (xe mới phát hiện), không chờ timeout."""
        self.end_capture(current_sec)

    def should_end_capture(self, current_sec: float) -> bool:
        """True nếu đã quá idle_timeout giây không thấy xe."""
        if self._state != EventState.CAPTURING:
            return False
        elapsed = current_sec - self._last_detection_sec
        return elapsed >= self._cfg.event_idle_timeout

    def end_capture(self, current_sec: float) -> None:
        """Chuyển từ CAPTURING → PROCESSING, ghi lại end_sec."""
        if self._event is not None:
            self._event.end_sec = current_sec
        self._state = EventState.PROCESSING

    def finish_processing(self) -> None:
        """Chuyển từ PROCESSING → IDLE, reset event."""
        self._state = EventState.IDLE
        self._event = None
        self._committed_plate = None
        self._committed_prefix = None

    # ── Cooldown helpers ──────────────────────────────────────────────────────

    def is_plate_in_cooldown(self, plate: str) -> bool:
        """True nếu biển số này vừa được gửi server chưa đủ cooldown."""
        last_sent = self._plate_cooldown.get(plate, 0.0)
        return (time.time() - last_sent) < self._cfg.event_cooldown

    def mark_plate_sent(self, plate: str) -> None:
        """Ghi nhận thời điểm gửi server cho biển số này."""
        self._plate_cooldown[plate] = time.time()
