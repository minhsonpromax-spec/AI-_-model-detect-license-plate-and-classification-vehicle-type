"""
Plate detection + OCR pipeline.

Refactor từ code gốc của project, tách thành class có thể reuse.
Yêu cầu:
    - helpers/utils_rotate.py  (deskew helper)
    - helpers/ocr.py           (read_plate helper)
    - ultralytics YOLO
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Import helpers từ project của user (đặt helpers/ cùng cấp với vehicle_pipeline/)
try:
    import helpers.utils_rotate as utils_rotate
    import helpers.ocr as ocr
except ImportError as exc:
    raise ImportError(
        "helpers/utils_rotate.py và helpers/ocr.py là bắt buộc. "
        "Đặt folder helpers/ cùng cấp với vehicle_pipeline/."
    ) from exc


class PlatePipeline:
    """
    Detect biển số xe bằng YOLO, đọc ký tự bằng OCR, voting để ra kết quả cuối.

    Cache kết quả theo track_id trong suốt event.
    Khi event kết thúc → gọi get_best_plate_all() → clear_cache().
    """

    def __init__(self, detect_model_path: str, ocr_model_path: str, config) -> None:
        self._cfg = config
        self._detector = YOLO(detect_model_path)
        self._ocr_model = YOLO(ocr_model_path)
        # track_id → list of (plate_text, confidence)
        self._cache: dict[int, list[tuple[str, float]]] = defaultdict(list)
        # Deque phẳng (mọi track_id) của các lần đọc gần nhất, để streak detection
        self._recent: deque[tuple[str, float]] = deque(maxlen=15)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_count: int) -> bool:
        """
        Chạy YOLO detect + OCR trên 1 frame.

        Returns:
            True nếu frame này có ít nhất 1 biển số được detect.
        """
        results = self._detector.track(
            frame,
            persist=True,
            conf=self._cfg.yolo_conf,
            verbose=False,
        )[0]

        if results.boxes is None:
            return False

        has_detection = False

        for box in results.boxes:
            if box.id is None:
                continue

            track_id = int(box.id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if w < self._cfg.plate_box_min_w or h < self._cfg.plate_box_min_h:
                continue

            has_detection = True

            # ── Vẽ bounding box lên frame (để debug hiển thị) ────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )

            # ── OCR mỗi 2 frame để giảm tải ─────────────────────────────────
            if frame_count % 2 != 0:
                continue

            crop = frame[y1:y2, x1:x2]
            lp, conf = self._read_plate(crop)

            if lp and lp != "unknown":
                self._cache[track_id].append((lp, conf))
                # Giữ history ngắn tránh stale data
                self._cache[track_id] = self._cache[track_id][-12:]
                # Flat queue cho streak detection (không phân biệt track_id)
                self._recent.append((lp, conf))

                cv2.putText(
                    frame, f"{lp} ({conf:.2f})",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (36, 255, 12), 2,
                )

            # Hiển thị best voted plate realtime
            best = self._vote(self._cache[track_id])
            if best:
                cv2.putText(
                    frame, f"[OK] {best}",
                    (x1, y1 - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
                )

        return has_detection

    def get_best_plate_all(self) -> Optional[str]:
        """
        Tổng hợp toàn bộ cache (mọi track_id) → voting → trả về biển số tốt nhất.
        Gọi khi event kết thúc.
        """
        all_entries: list[tuple[str, float]] = []
        for entries in self._cache.values():
            all_entries.extend(entries)
        return self._vote(all_entries)

    @staticmethod
    def _plate_prefix(text: str) -> str:
        """
        Lấy phần tỉnh/serie ổn định của biển số làm key so sánh.
        '75AA-37010' → '75AA' | '74G1-20202' → '74G1' | '51A-12345' → '51A'
        """
        for sep in ("-", " ", "."):
            if sep in text:
                return text.split(sep)[0].strip()
        # Không có dấu phân cách: lấy 4 ký tự đầu
        return text[:4]

    def get_dominant_prefix(self, min_streak: int = 3) -> Optional[str]:
        """
        Trả về prefix nếu `min_streak` lần đọc CUỐI CÙNG liên tiếp đều cùng prefix.
        None nếu chưa đủ dữ liệu hoặc không nhất quán.

        Dùng prefix ('75AA', '74G1'...) thay vì full text để chịu được OCR noise
        trên phần số cuối biển ('75AA-37010' vs '75AA-37012' đều → prefix '75AA').
        """
        if len(self._recent) < min_streak:
            return None
        tail_prefixes = [
            self._plate_prefix(t) for t, _ in list(self._recent)[-min_streak:]
        ]
        if len(set(tail_prefixes)) == 1:
            return tail_prefixes[0]
        return None

    def clear_cache(self) -> None:
        """Xóa toàn bộ cache. Gọi sau khi event được xử lý xong."""
        self._cache.clear()
        self._recent.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_plate(self, crop: np.ndarray) -> tuple[str, float]:
        """Thử deskew các góc xoay khác nhau, lấy kết quả OCR đầu tiên hợp lệ."""
        for cc in range(2):
            for ct in range(2):
                rotated = utils_rotate.deskew(crop, cc, ct)
                lp, conf = ocr.read_plate(self._ocr_model, rotated)
                logger.debug("[OCR] cc=%d ct=%d → '%s' conf=%.2f", cc, ct, lp, conf)
                if lp != "unknown":
                    return lp, conf
        return "unknown", 0.0

    def _vote(self, plates: list[tuple[str, float]]) -> Optional[str]:
        """
        Weighted voting: tổng confidence theo text.

        2 cơ chế pass:
        ① Normal  : tổng score >= plate_min_score VÀ ratio >= plate_min_ratio
                    → xe đi vừa hoặc chậm, đọc được nhiều lần
        ② Fast-car: chỉ 1-2 lần đọc nhưng conf cao (>= plate_fast_conf)
                    VÀ ratio >= 0.90 (gần như tất cả đọc ra cùng 1 biển)
                    → xe đi nhanh qua cổng
        """
        if not plates:
            return None

        score_map: dict[str, float] = defaultdict(float)
        best_single_conf: dict[str, float] = {}  # conf cao nhất cho mỗi biển

        for text, conf in plates:
            score_map[text] += conf
            if conf > best_single_conf.get(text, 0.0):
                best_single_conf[text] = conf

        best = max(score_map, key=score_map.get)
        best_score = score_map[best]
        total_score = sum(score_map.values())
        ratio = best_score / total_score if total_score > 0 else 0.0

        # ① Normal voting
        if best_score >= self._cfg.plate_min_score and ratio >= self._cfg.plate_min_ratio:
            logger.debug(
                "[Vote] ✓ Normal  '%s'  score=%.2f  ratio=%.0f%%",
                best, best_score, ratio * 100,
            )
            return best

        # ② Fast-car fallback
        fast_conf = getattr(self._cfg, "plate_fast_conf", 0.80)
        if best_single_conf.get(best, 0.0) >= fast_conf and ratio >= 0.90:
            logger.debug(
                "[Vote] ✓ FastCar '%s'  best_conf=%.2f  ratio=%.0f%%",
                best, best_single_conf[best], ratio * 100,
            )
            return best

        logger.debug(
            "[Vote] ✗ Reject  '%s'  score=%.2f/%.2f  ratio=%.0f%%/%.0f%%",
            best, best_score, self._cfg.plate_min_score,
            ratio * 100, self._cfg.plate_min_ratio * 100,
        )
        return None
