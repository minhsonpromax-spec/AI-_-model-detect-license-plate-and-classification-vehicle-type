from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Thêm root của audio_model vào sys.path để import được api.audio_engine
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# parents[1] là thư mục cha của thư mục chứa code đó, tức là thư mục cha của thư mục vehicle_pipeline

from audio_module import AudioEngine

from vehicle_pipeline.api_client import ApiClient
from vehicle_pipeline.audio_extractor import extract_audio_segment
from vehicle_pipeline.config import PipelineConfig
from vehicle_pipeline.event_manager import EventManager, EventState
from vehicle_pipeline.plate_pipeline import PlatePipeline
from vehicle_pipeline.vision_classifier import RealVisionClassifier


def _effective_bypass_ocsvm(config: PipelineConfig) -> bool:
    """Giống logic trong run(): CLI `--bypass-ocsvm` nếu có, không thì đọc từ audio_config YAML."""
    ovr = getattr(config, "_audio_cfg_override", None)
    if ovr is not None:
        return bool(ovr.bypass_ocsvm)
    from audio_module.config.config import AudioModuleConfig

    return bool(AudioModuleConfig.from_yaml(config.audio_config).bypass_ocsvm)


@dataclass
class EventStats:
    """Kết quả của một event để in thống kê cuối."""
    plate: str
    duration_sec: float
    vision_gas: float
    vision_elec: float
    vision_frames: int
    ocsvm_inlier: int       # số cửa sổ inlier
    ocsvm_outlier: int      # số cửa sổ outlier
    cnn_gas_mean: float     # trung bình gasoline prob từ CNN (chỉ inlier khi bypass=False)
    cnn_gas_bypass: float   # trung bình gasoline prob khi bypass (mọi cửa sổ)
    audio_gate: float
    entropy: float
    fusion_label: str
    fusion_gas: float
    fusion_elec: float
    api_ok: bool


def _build_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Tắt debug log của các lib bên ngoài, chỉ giữ debug của pipeline
    for noisy in ("ultralytics", "tensorflow", "absl", "matplotlib", "PIL",
                  "numba", "numba.core", "numba.core.byteflow", "h5py",
                  "librosa", "audioread"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("vehicle_pipeline")


def run(config: PipelineConfig) -> None:
    log = _build_logger()

    # ── Load tất cả models ────────────────────────────────────────────────────
    log.info("Loading models…")

    plate_pipeline = PlatePipeline(
        detect_model_path=config.yolo_lp_detect,
        ocr_model_path=config.yolo_lp_ocr,
        config=config,
    )

    # Vision classifier — dùng model detect_type_vehicle.pt
    vision_clf = RealVisionClassifier(config.vision_model_path)

    audio_engine = AudioEngine.from_yaml(config.audio_config, logger=log)

    # Nếu CLI truyền --bypass-ocsvm, override config của audio_engine
    if hasattr(config, "_audio_cfg_override"):
        audio_engine._pipeline._cfg = config._audio_cfg_override
        audio_engine._cfg = config._audio_cfg_override
        log.info("[Config] bypass_ocsvm overridden → %s", config._audio_cfg_override.bypass_ocsvm)
    api_client = ApiClient(config, app_logger=log)
    event_manager = EventManager(config)

    log.info("Models loaded.")

    # collector dùng để in thống kê cuối video
    stats_list: List[EventStats] = []

    # ── Mở video ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {config.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        log.warning("Không đọc được FPS từ video, dùng mặc định %.1f", fps)

    log.info("Video: %s  |  FPS=%.1f  |  Press Q to quit", config.video_path, fps)

    frame_count = 0
    # vẫn dùng cho real time đc

    # ════════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            # Hết video — nếu đang capturing thì kết thúc event luôn
            if event_manager.state == EventState.CAPTURING:
                current_sec = frame_count / fps
                event_manager.end_capture(current_sec)
                # Xử lý event cuối trước khi thoát
                _process_event(
                    event=event_manager.current_event,
                    config=config,
                    plate_pipeline=plate_pipeline,
                    audio_engine=audio_engine,
                    api_client=api_client,
                    event_manager=event_manager,
                    log=log,
                    stats_list=stats_list,
                )
            break

        frame_count += 1
        current_sec = frame_count / fps

        # Chỉ xử lý 1/N frame để tăng hiệu suất
        if frame_count % config.frame_skip != 0:
            _show_frame(frame, event_manager)
            continue

        # ── Bỏ qua detection khi đang process event trước ────────────────────
        if event_manager.state == EventState.PROCESSING:
            _show_frame(frame, event_manager)
            continue

        # ── YOLO detect + OCR biển số ─────────────────────────────────────────
        has_vehicle = plate_pipeline.process_frame(frame, frame_count)

        if has_vehicle:
            event_manager.on_detection(current_sec)

            # Thu thập vision scores trong lúc capturing
            if event_manager.state == EventState.CAPTURING:
                vision_score = vision_clf.predict(frame)
                event_manager.current_event.vision_scores.append(vision_score)

                # ── Commit plate đại diện cho event (lần đầu có đủ dữ liệu vote) ─
                if event_manager._committed_plate is None:
                    rt_plate = plate_pipeline.get_best_plate_all()
                    if rt_plate:
                        prefix = plate_pipeline._plate_prefix(rt_plate)
                        event_manager.commit_plate(rt_plate, prefix=prefix)
                        log.debug("[Event] Committed plate: %s (prefix=%s)", rt_plate, prefix)

                # ── Phát hiện xe MỚI qua dominant prefix trong rolling window ─
                dominant = plate_pipeline.get_dominant_prefix(min_streak=3)
                if dominant and event_manager.detect_new_vehicle(dominant):
                    log.info(
                        "[Event] ✦ Xe MỚI phát hiện! Prefix '%s' ≠ committed '%s' "
                        "→ kết thúc event cũ (t=%.2fs)",
                        dominant, event_manager._committed_prefix, current_sec,
                    )
                    event_manager.force_end_capture(current_sec)

        # ── Kiểm tra kết thúc event (timeout) ────────────────────────────────
        if event_manager.should_end_capture(current_sec):
            event_manager.end_capture(current_sec)

        # ── Xử lý event vừa kết thúc ─────────────────────────────────────────
        if event_manager.state == EventState.PROCESSING:
            _process_event(
                event=event_manager.current_event,
                config=config,
                plate_pipeline=plate_pipeline,
                audio_engine=audio_engine,
                api_client=api_client,
                event_manager=event_manager,
                log=log,                stats_list=stats_list,            )
            event_manager.finish_processing()

        _show_frame(frame, event_manager)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Người dùng dừng pipeline.")
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Pipeline kết thúc.")
    _print_stats(stats_list, log, config)


# ════════════════════════════════════════════════════════════════════════════════
# PROCESS EVENT
# ════════════════════════════════════════════════════════════════════════════════

def _process_event(
    event,
    config: PipelineConfig,
    plate_pipeline: PlatePipeline,
    audio_engine: AudioEngine,
    api_client: ApiClient,
    event_manager: EventManager,
    log: logging.Logger,
    stats_list: List[EventStats] | None = None,
) -> None:

    sep = "─" * 55
    log.info(sep)
    log.info("[Event] t=[%.2fs → %.2fs]  duration=%.2fs",
             event.start_sec, event.end_sec, event.end_sec - event.start_sec)

    # ── 1. Lấy biển số tốt nhất ───────────────────────────────────────────────
    plate = plate_pipeline.get_best_plate_all()
    plate_pipeline.clear_cache()

    if plate is None:
        log.warning("[Event] ✗ Biển số: KHÔNG XÁC ĐỊNH ĐƯỢC → bỏ qua event.")
        log.info(sep)
        return

    log.info("[Event] ✓ Biển số: %s", plate)

    # ── 2. Cooldown check ─────────────────────────────────────────────────────
    if event_manager.is_plate_in_cooldown(plate):
        log.info("[Event] ⏸ %s đang trong cooldown → bỏ qua (đã gửi gần đây).", plate)
        log.info(sep)
        return

    # ── 3. Vision summary ─────────────────────────────────────────────────────
    if event.vision_scores:
        vision_summary = {
            "gasoline": float(np.mean([s["gasoline"] for s in event.vision_scores])),
            "electric": float(np.mean([s["electric"] for s in event.vision_scores])),
        }
        log.info("[Event] Vision (%d frames): gasoline=%.3f  electric=%.3f",
                 len(event.vision_scores),
                 vision_summary["gasoline"], vision_summary["electric"])
    else:
        vision_summary = {"gasoline": 0.5, "electric": 0.5}
        log.info("[Event] Vision: không có frames → dùng 50/50")

    # ── 4. Extract audio + chạy AudioEngine ──────────────────────────────────
    vehicle_type = "uncertain"
    fusion_result: dict = {}

    try:
        waveform = extract_audio_segment(
            video_path=config.video_path,
            start_sec=event.start_sec,
            end_sec=event.end_sec,
            sample_rate=16000,
            temp_dir=config.audio_temp_dir,
            keep_wav=config.keep_debug_audio,
            debug_audio_dir=config.debug_audio_dir,
        )
        log.info("[Event] Audio: %.2f giây  (%d samples)", waveform.size / 16000, waveform.size)

        fusion_result = audio_engine.process(
            waveform=waveform,
            sample_rate=16000,
            vision_summary=vision_summary,
            event_timestamp=event.start_sec,
        )
        vehicle_type = fusion_result.get("final_label", "uncertain")

        log.info(
            "[Event] Fusion: gas=%.3f  elec=%.3f  audio_gate=%.1f  entropy=%.3f",
            fusion_result.get("score_gasoline", 0),
            fusion_result.get("score_electric", 0),
            fusion_result.get("audio_gate", 0),
            fusion_result.get("entropy", 1),
        )

    except Exception as exc:
        log.error("[Event] Audio/fusion lỗi: %s → fallback về vision only.", exc)
        vehicle_type = "gasoline" if vision_summary["gasoline"] >= vision_summary["electric"] else "electric"

    # ── 5. Kết quả cuối ───────────────────────────────────────────────────────
    type_label = {"gasoline": "XE XĂNG 🔥", "electric": "XE ĐIỆN ⚡", "uncertain": "KHÔNG RÕ ❓"}.get(
        vehicle_type, vehicle_type.upper()
    )
    log.info("┌─────────────────────────────────────────┐")
    log.info("│  BIỂN SỐ  : %-28s │", plate)
    log.info("│  LOẠI XE  : %-28s │", type_label)
    log.info("└─────────────────────────────────────────┘")

    # ── 6. Gửi lên server ─────────────────────────────────────────────────────
    success = api_client.send_vehicle_event(
        plate=plate,
        vehicle_type=vehicle_type,
        extra=fusion_result or None,
    )

    if success:
        event_manager.mark_plate_sent(plate)
        log.info("[API] ✓ Đã gửi server thành công")
    else:
        log.error("[API] ✗ Gửi server THẤT BẠI (sẽ không thử lại)")

    # ── 7. Ghi stats ─────────────────────────────────────────────────────────
    if stats_list is not None:
        # Lấy per-window OCSVM/CNN detail từ audio_engine nếu có
        _w = getattr(audio_engine, "_last_window_results", [])
        n_inlier  = sum(1 for r in _w if getattr(r, "ocsvm_label", "") == "inlier")
        n_outlier = sum(1 for r in _w if getattr(r, "ocsvm_label", "") == "outlier")
        # gasoline prob trung bình từ tất cả cửa sổ (bypass mode)
        gas_all  = [getattr(r, "gasoline_prob", 0.0) for r in _w]
        gas_inlier = [getattr(r, "gasoline_prob", 0.0) for r in _w
                      if getattr(r, "ocsvm_label", "") == "inlier"]
        stats_list.append(EventStats(
            plate=plate,
            duration_sec=round(event.end_sec - event.start_sec, 2),
            vision_gas=round(vision_summary["gasoline"], 3),
            vision_elec=round(vision_summary["electric"], 3),
            vision_frames=len(event.vision_scores),
            ocsvm_inlier=n_inlier,
            ocsvm_outlier=n_outlier,
            cnn_gas_mean=round(float(np.mean(gas_inlier)) if gas_inlier else 0.0, 3),
            cnn_gas_bypass=round(float(np.mean(gas_all)) if gas_all else 0.0, 3),
            audio_gate=round(fusion_result.get("audio_gate", 0.0), 2),
            entropy=round(fusion_result.get("entropy", 1.0), 3),
            fusion_label=vehicle_type,
            fusion_gas=round(fusion_result.get("score_gasoline", 0.0), 3),
            fusion_elec=round(fusion_result.get("score_electric", 0.0), 3),
            api_ok=success,
        ))

    log.info(sep)


# ── Stats printer ─────────────────────────────────────────────────────────────

def _print_stats(
    stats_list: List[EventStats],
    log: logging.Logger,
    config: PipelineConfig,
) -> None:
    if not stats_list:
        log.info("[Stats] Không có event nào được xử lý.")
        return

    is_bypass = _effective_bypass_ocsvm(config)
    bypass_tag = "ON" if is_bypass else "OFF"

    W = 70
    div  = "═" * W
    thin = "─" * W

    log.info("")
    log.info(div)
    log.info(
        "  THỐNG KÊ KẾT QUẢ PIPELINE  (bypass_ocsvm=%s)",
        bypass_tag,
    )
    log.info(div)

    for i, s in enumerate(stats_list, 1):
        log.info("  Event #%d", i)
        log.info(thin)
        log.info("  Biển số     : %s", s.plate)
        log.info("  Thời lượng  : %.2f s", s.duration_sec)
        log.info("")
        log.info("  [Vision]")
        log.info("    Frames      : %d", s.vision_frames)
        log.info("    Xăng        : %.3f   |  Điện: %.3f", s.vision_gas, s.vision_elec)
        log.info("")
        log.info("  [Audio - OCSVM]")
        log.info("    Inlier      : %d cửa sổ", s.ocsvm_inlier)
        log.info("    Outlier     : %d cửa sổ  (bình thường nếu audio từ mp4)", s.ocsvm_outlier)
        log.info("  [Audio - CNN]  (bypass=%s)", bypass_tag)
        log.info("    Gas (all)   : %.3f  ← mọi cửa sổ (khi bypass=ON, CNN chạy cả outlier)", s.cnn_gas_bypass)
        log.info("    Gas (inlier): %.3f  ← chỉ inlier (khớp aggregate/Fusion khi bypass=OFF)", s.cnn_gas_mean)
        log.info("    Audio gate  : %.1f   |  Entropy: %.3f", s.audio_gate, s.entropy)
        log.info("")
        log.info("  [Fusion]")
        log.info("    Score xăng  : %.3f   |  Score điện: %.3f", s.fusion_gas, s.fusion_elec)
        log.info("    Kết quả     : %s", s.fusion_label.upper())
        log.info("    API         : %s", "✓ OK" if s.api_ok else "✗ FAILED")
        if i < len(stats_list):
            log.info("")

    log.info(div)
    log.info("  Tổng: %d event  |  Xăng: %d  |  Điện: %d  |  Không rõ: %d",
             len(stats_list),
             sum(1 for s in stats_list if s.fusion_label == "gasoline"),
             sum(1 for s in stats_list if s.fusion_label == "electric"),
             sum(1 for s in stats_list if s.fusion_label == "uncertain"))
    log.info(div)
    log.info("")


# ── Debug display ─────────────────────────────────────────────────────────────

def _show_frame(frame: np.ndarray, event_manager: EventManager) -> None:
    state_colors = {
        EventState.IDLE:       (100, 100, 100),
        EventState.CAPTURING:  (0, 200, 0),
        EventState.PROCESSING: (0, 165, 255),
    }
    color = state_colors.get(event_manager.state, (255, 255, 255))

    cv2.putText(
        frame, f"State: {event_manager.state.name}",
        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2,
    )
    cv2.imshow("Vehicle Pipeline", frame)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vehicle Entry Pipeline")
    parser.add_argument("--video",         default="video/test.mp4",   help="Đường dẫn file mp4")
    parser.add_argument("--audio-config",  default="audio_module/config/config.yaml", help="Config audio_model")
    parser.add_argument("--api-url",       default="https://aibuildingmanager.online")
    parser.add_argument("--bypass-ocsvm",  default=None, choices=["true", "false"],
                        help="Override bypass_ocsvm trong config (true=debug mp4, false=production mic)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        video_path=args.video,
        audio_config=args.audio_config,
        api_url=args.api_url,
    )

    # Override bypass_ocsvm nếu truyền từ CLI (không sửa file yaml)
    if args.bypass_ocsvm is not None:
        from audio_module.config.config import AudioModuleConfig
        _audio_cfg = AudioModuleConfig.from_yaml(cfg.audio_config)
        _audio_cfg = AudioModuleConfig(
            **{**_audio_cfg.__dict__, "bypass_ocsvm": args.bypass_ocsvm == "true"}
        )
        # Patch vào PipelineConfig để run() dùng
        cfg._audio_cfg_override = _audio_cfg

    run(cfg)

