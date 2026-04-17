"""
Extract một đoạn audio [start_sec, end_sec] từ file mp4 → numpy array.

Strategy:
    1. Dùng ffmpeg subprocess để cắt đúng khoảng thời gian → file .wav tạm
    2. librosa.load() đọc .wav → numpy float32
    3. Xóa file .wav tạm sau khi đọc xong

Yêu cầu: ffmpeg phải được cài và có trong PATH.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
    temp_dir: str = "temp_audio",
    keep_wav: bool = False,          # True → giữ file .wav lại để nghe debug
    debug_audio_dir: str = "debug_audio",  # thư mục lưu khi keep_wav=True
) -> np.ndarray:
    """
    Extract audio segment [start_sec, end_sec] từ mp4 bằng ffmpeg.

    Args:
        video_path : Đường dẫn file mp4.
        start_sec  : Thời điểm bắt đầu (giây từ đầu video).
        end_sec    : Thời điểm kết thúc (giây từ đầu video).
        sample_rate: Sample rate đầu ra (mặc định 16kHz — khớp AudioEngine).
        temp_dir   : Thư mục lưu file .wav tạm.

    Returns:
        numpy float32 array shape (N,) — mono waveform.

    Raises:
        RuntimeError : Nếu ffmpeg thất bại.
        FileNotFoundError: Nếu video_path không tồn tại.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required: pip install librosa") from exc

    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"end_sec ({end_sec:.2f}) must be > start_sec ({start_sec:.2f})")

    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Tên file theo khoảng thời gian — đọc được khi nghe debug
    seg_name = f"seg_{start_sec:.2f}s_to_{end_sec:.2f}s.wav"
    tmp_wav = os.path.join(temp_dir, seg_name)

    # ── Gọi ffmpeg (không dùng shell=True để tránh command injection) ────────
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",         # seek trước -i để nhanh hơn
        "-i", video_path,
        "-t", f"{duration:.6f}",
        "-ar", str(sample_rate),            # resample về target sr
        "-ac", "1",                         # mono
        "-f", "wav",
        tmp_wav,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and add it to PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while extracting audio.") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}:\n{err_msg}")

    # ── Load wav → numpy ─────────────────────────────────────────────────────
    try:
        waveform, _ = librosa.load(tmp_wav, sr=sample_rate, mono=True)
    finally:
        if keep_wav:
            # Copy sang debug_audio/ với tên đẹp hơn trước khi xóa tmp
            Path(debug_audio_dir).mkdir(parents=True, exist_ok=True)
            debug_path = os.path.join(debug_audio_dir, seg_name)
            try:
                import shutil
                shutil.copy2(tmp_wav, debug_path)
                logger.info("[AudioExtractor] Saved debug audio → %s", debug_path)
            except OSError as e:
                logger.warning("[AudioExtractor] Không lưu được debug audio: %s", e)
            # Xóa file tạm
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
        else:
            # Xóa ngay file tạm
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    if waveform.size == 0:
        logger.warning(
            "[AudioExtractor] Empty waveform for segment [%.2f, %.2f]",
            start_sec, end_sec,
        )

    return waveform.astype(np.float32)
