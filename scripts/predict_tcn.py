# -*- coding: utf-8 -*-
"""Predict funscript position values using the TCN model.

Usage:
    # Predict from a scene that already has features extracted:
    python scripts/predict_tcn.py --scene scene_00018_t00926_40s

    # Predict directly from an mp4 video:
    python scripts/predict_tcn.py --video path/to/video.mp4 --out out.funscript

    # For VR side-by-side video (default):
    python scripts/predict_tcn.py --video video.mp4 --vr --sbs-crop left

    # For flat (non-VR) video:
    python scripts/predict_tcn.py --video video.mp4 --no-vr

    # Use a specific checkpoint:
    python scripts/predict_tcn.py --scene scene_00018_t00926_40s --checkpoint data/models/checkpoints_tcn/best_tcn.pt

    # Show a matplotlib plot comparing prediction to ground-truth labels:
    python scripts/predict_tcn.py --scene scene_00018_t00926_40s --plot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.tcn import FunscriptTCN
from src.data.pose import load_pose_model
from src.data.extraction import SinglePassExtractor, extract_single_pass_batched
from src.data.decode import stream_video_gpu
from src.data.flow import compute_flow_raft_batched

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

KP_FILE = "keypoints/pose-vrlens-finetunes-large.npy"
EMB_FILE = "embeddings/pose-vrlens-finetunes-large.npy"
KP_FILE_MULTICLASS = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
EMB_FILE_MULTICLASS = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
FLOW_FILE = "flow/raft_f64_s0.5.npy"


def _extract_pose_batched(
    yolo_model,
    frames_rgb: list[np.ndarray],
    max_persons: int,
    n_frames: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract pose keypoints in batches using Ultralytics predict."""
    all_kpts = []
    for i in range(0, n_frames, batch_size):
        batch = np.stack(frames_rgb[i : i + batch_size])
        kpts = extract_pose_batch(
            yolo_model, batch, max_persons=max_persons, n_keypoints=21,
        )
        all_kpts.append(kpts)
    return np.concatenate(all_kpts, axis=0)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FunscriptTCN, dict]:
    """Load TCN model from checkpoint. Returns (model, model_config)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = FunscriptTCN(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    is_mc = cfg.get("n_partners") is not None
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.6f}"
          f" [{'multiclass' if is_mc else 'single-class'}]")
    return model, cfg


def load_stats(data_dir: Path, n_persons: int = 10, embed_dim: int = 512, flow_dim: int = 64):
    stats_path = data_dir / "feature_stats.npz"
    if not stats_path.exists():
        print("Warning: no feature_stats.npz — predictions will use un-normalized features")
        return None, None, None, None

    stats = np.load(stats_path)
    emb_mean = emb_std = None
    expected = n_persons * embed_dim
    if "emb_mean" in stats and stats["emb_mean"].shape[0] == expected:
        emb_mean = stats["emb_mean"].reshape(n_persons, embed_dim)
        emb_std = stats["emb_std"].reshape(n_persons, embed_dim)
    flow_mean = flow_std = None
    if "flow_mean" in stats and stats["flow_mean"].shape[0] == flow_dim:
        flow_mean = stats["flow_mean"]
        flow_std = stats["flow_std"]
    return emb_mean, emb_std, flow_mean, flow_std


def _probe_video(video_path: Path) -> tuple[int, int, str]:
    """Return (width, height, codec_name) via ffprobe. Raises on failure."""
    import subprocess, json as _json
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-of", "json", str(video_path),
    ]
    probe = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    info = _json.loads(probe.stdout)["streams"][0]
    return int(info["width"]), int(info["height"]), info.get("codec_name", "")


def _build_ffmpeg_cmd(
    video_path: Path,
    src_w: int,
    src_h: int,
    codec_name: str,
    target_size: int | None,
    crop_left_half: bool,
    start_time: float | None,
    duration: float | None,
    target_fps: float | None,
) -> tuple[list[str], int, int]:
    """Build ffmpeg command and return (cmd, out_w, out_h)."""
    import subprocess
    cmd: list[str] = ["ffmpeg", "-v", "error"]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    hw_decoder = None
    if codec_name in ("hevc", "h265"):
        hw_decoder = "hevc_cuvid"
    elif codec_name in ("h264", "avc"):
        hw_decoder = "h264_cuvid"

    use_cuvid_crop_resize = hw_decoder is not None and target_size is not None

    if hw_decoder:
        cmd.extend(["-hwaccel", "cuda", "-c:v", hw_decoder])
        if use_cuvid_crop_resize:
            decoder_opts = []
            if crop_left_half:
                crop_right = src_w // 2
                decoder_opts.append(f"crop=0x0x0x{crop_right}")
            decoder_opts.append(f"resize={target_size}x{target_size}")
            for opt in decoder_opts:
                cmd.extend(["-" + opt.split("=")[0], opt.split("=")[1]])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend(["-i", str(video_path)])

    vf_parts: list[str] = []
    if not use_cuvid_crop_resize:
        if crop_left_half:
            vf_parts.append(f"crop={src_w // 2}:{src_h}:0:0")
        if target_size is not None:
            vf_parts.append(f"scale={target_size}:{target_size}")
    if target_fps is not None:
        vf_parts.append(f"fps={target_fps}")

    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    if use_cuvid_crop_resize:
        out_w = out_h = target_size
    else:
        out_w = target_size or (src_w // 2 if crop_left_half else src_w)
        out_h = target_size or src_h

    cmd.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return cmd, out_w, out_h


def _stream_frames_ffmpeg(
    video_path: Path,
    target_size: int | None = None,
    crop_left_half: bool = False,
    start_time: float | None = None,
    duration: float | None = None,
    target_fps: float | None = None,
    chunk_size: int = 240,
):
    """Yield chunks of RGB uint8 frames via ffmpeg NVDEC. Falls back to cv2 on error.

    Each chunk is a list of np.ndarray frames. Chunks are processed and freed
    immediately to avoid holding the full video in RAM.
    """
    import subprocess

    try:
        src_w, src_h, codec_name = _probe_video(video_path)
    except Exception:
        yield from _stream_frames_cv2(video_path, chunk_size)
        return

    cmd, out_w, out_h = _build_ffmpeg_cmd(
        video_path, src_w, src_h, codec_name,
        target_size, crop_left_half, start_time, duration, target_fps,
    )

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = out_w * out_h * 3
        chunk: list[np.ndarray] = []
        total = 0
        while True:
            data = proc.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
            chunk.append(frame)
            total += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        proc.wait()
        if proc.returncode != 0 and total == 0:
            stderr = proc.stderr.read().decode(errors="replace")
            if stderr:
                print(f"  ffmpeg stderr: {stderr[:200]}")
            yield from _stream_frames_cv2(video_path, chunk_size)
            return
        if chunk:
            yield chunk
    except Exception as e:
        print(f"  ffmpeg decode error: {e}")
        yield from _stream_frames_cv2(video_path, chunk_size)


def _stream_frames_cv2(video_path: Path, chunk_size: int = 512):
    """Yield chunks of RGB uint8 frames via OpenCV CPU decode."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    chunk: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        chunk.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    cap.release()
    if chunk:
        yield chunk


def extract_features_from_video(
    video_path: Path,
    pose_model_path: Path,
    vr_mode: bool,
    sbs_crop: str,
    frame_size: int,
    max_persons: int,
    device: torch.device,
    start_time: float | None = None,
    duration: float | None = None,
    target_fps: float | None = None,
    chunk_size: int = 512,
    multiclass: bool = False,
    max_partners: int = 5,
    max_beholders: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract keypoints, embeddings, and flow features from a raw video.

    Uses single-pass YOLO extraction (pose + RoI Align embeddings from one
    forward pass) and torchcodec GPU decode for maximum throughput.
    """
    timings: dict[str, float] = {
        "model_load": 0.0, "video_decode": 0.0,
        "yolo_single_pass": 0.0, "raft_flow": 0.0,
    }

    # ── Load pose model + single-pass extractor ──────────────────────────
    t0 = time.perf_counter()
    print(f"Loading pose model from {pose_model_path}...")
    pose_model = load_pose_model(
        model_name="yolo11m-pose",
        model_path=str(pose_model_path),
        device=str(device),
    )
    extractor = SinglePassExtractor(
        pose_model, max_persons=max_persons, n_keypoints=21,
        confidence_threshold=0.02, device=str(device),
        multiclass=multiclass,
        max_partners=max_partners,
        max_beholders=max_beholders,
    )
    timings["model_load"] = time.perf_counter() - t0

    batch_size = 32
    flow_size = 320
    crop_left_half = vr_mode and sbs_crop == "left"

    all_keypoints: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_flow: list[np.ndarray] = []
    prev_flow_frame: np.ndarray | None = None
    n_frames = 0

    # Estimate total frames for progress bar
    total_est = None
    try:
        from torchcodec.decoders import VideoDecoder as _VD
        _tmp = _VD(str(video_path))
        src_fps = _tmp.metadata.average_fps or 30.0
        src_n = _tmp.metadata.num_frames or 0
        _tmp = None
        eff_fps = target_fps or src_fps
        if duration:
            total_est = int(duration * eff_fps)
        elif src_fps > 0:
            total_est = int(src_n * eff_fps / src_fps)
    except Exception:
        pass

    tp = tqdm(
        total=max(1, total_est // chunk_size) if total_est else None,
        unit=f"×{chunk_size}f", desc="Processing", dynamic_ncols=True,
    )

    t_last = time.perf_counter()

    for chunk_data in stream_video_gpu(
        video_path,
        device=str(device),
        crop_left_half=crop_left_half,
        target_size=frame_size,
        target_fps=target_fps,
        start_time=start_time,
        duration=duration,
        chunk_size=chunk_size,
        as_numpy=False,
    ):
        t_decode_end = time.perf_counter()
        timings["video_decode"] += t_decode_end - t_last

        # Convert to numpy for YOLO compatibility
        if isinstance(chunk_data, torch.Tensor):
            chunk_np = chunk_data.cpu().numpy()
        elif isinstance(chunk_data, list):
            chunk_np = np.stack(chunk_data)
        else:
            chunk_np = chunk_data

        chunk_n = len(chunk_np)
        n_frames += chunk_n
        pose_frames = list(chunk_np)

        # Resize for flow (320×320)
        flow_frames = [cv2.resize(f, (flow_size, flow_size)) for f in pose_frames]

        # ── Single-pass YOLO: keypoints + embeddings ─────────────────────
        t0 = time.perf_counter()
        kp, emb = extract_single_pass_batched(extractor, pose_frames, batch_size)
        all_keypoints.append(kp)
        all_embeddings.append(emb)
        timings["yolo_single_pass"] += time.perf_counter() - t0

        del pose_frames

        # ── RAFT flow (with cross-chunk continuity) ──────────────────────
        t0 = time.perf_counter()
        if prev_flow_frame is not None:
            flow_input = np.stack([prev_flow_frame] + flow_frames)
            flow_chunk = compute_flow_raft_batched(
                flow_input, output_features=64, device=str(device), batch_size=64,
            )
            all_flow.append(flow_chunk[1:])
        else:
            flow_input = np.stack(flow_frames)
            flow_chunk = compute_flow_raft_batched(
                flow_input, output_features=64, device=str(device), batch_size=64,
            )
            all_flow.append(flow_chunk)
        prev_flow_frame = flow_frames[-1]
        del flow_frames
        timings["raft_flow"] += time.perf_counter() - t0

        t_last = time.perf_counter()
        tp.update(1)

    tp.close()
    extractor.close()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  Processed {n_frames} frames total")

    # ── Concatenate all chunks ───────────────────────────────────────────
    keypoints = np.concatenate(all_keypoints, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
    flow = np.concatenate(all_flow, axis=0)

    # ── Align frame counts ───────────────────────────────────────────────
    n = min(len(keypoints), len(embeddings), len(flow))
    keypoints = keypoints[:n]
    embeddings = embeddings[:n]
    flow = flow[:n]

    # ── Timing summary ───────────────────────────────────────────────────
    total = sum(timings.values())
    print(f"\n  Timing breakdown:")
    for k, v in timings.items():
        print(f"    {k:20s}: {v:6.2f}s ({v / total * 100:4.1f}%)")
    print(f"    {'TOTAL':20s}: {total:6.2f}s")

    return keypoints, embeddings, flow


def sliding_window_predict(
    model: FunscriptTCN,
    keypoints: np.ndarray,
    embeddings: np.ndarray,
    flow: np.ndarray,
    device: torch.device,
    seq_len: int = 120,
    stride: int = 60,
) -> np.ndarray:
    """Slide a window over the full sequence and average overlapping predictions."""
    n_frames = len(keypoints)
    pred_sum = np.zeros(n_frames, dtype=np.float32)
    pred_count = np.zeros(n_frames, dtype=np.float32)

    starts = list(range(0, n_frames - seq_len + 1, stride))
    # Always include a window ending at the last frame
    if n_frames >= seq_len and starts[-1] + seq_len < n_frames:
        starts.append(n_frames - seq_len)

    with torch.no_grad():
        for start in starts:
            end = start + seq_len
            kp = torch.from_numpy(keypoints[start:end]).float().unsqueeze(0).to(device)
            emb = torch.from_numpy(embeddings[start:end]).float().unsqueeze(0).to(device)
            fl = torch.from_numpy(flow[start:end]).float().unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(kp, emb, fl)  # [1, seq_len]

            p = out[0].float().cpu().numpy()

            # Triangular weighting: higher weight toward center
            weight = np.bartlett(seq_len).astype(np.float32) + 0.01
            pred_sum[start:end] += p * weight
            pred_count[start:end] += weight

    # Pad the beginning if seq_len > n_frames (very short clip)
    if n_frames < seq_len:
        kp = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)
        emb = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
        fl = torch.from_numpy(flow).float().unsqueeze(0).to(device)
        # Pad to seq_len
        pad = seq_len - n_frames
        kp = torch.nn.functional.pad(kp, (0, 0, 0, 0, 0, 0, 0, pad))
        emb = torch.nn.functional.pad(emb, (0, 0, 0, 0, 0, pad))
        fl = torch.nn.functional.pad(fl, (0, 0, 0, pad))
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(kp, emb, fl)
        return out[0, :n_frames].float().cpu().numpy()

    # Normalize by overlap count
    mask = pred_count > 0
    pred_sum[mask] /= pred_count[mask]

    return pred_sum


def predictions_to_funscript(positions: np.ndarray, fps: float = 30.0, start_time: float = 0.0) -> dict:
    """Convert per-frame position array [0,1] to funscript JSON format."""
    actions = []
    for frame_idx, pos in enumerate(positions):
        at_ms = int(round((frame_idx / fps + start_time) * 1000.0))
        pos_int = int(round(float(pos) * 100.0))
        pos_int = max(0, min(100, pos_int))
        actions.append({"at": at_ms, "pos": pos_int})

    return {
        "version": "1.0",
        "inverted": False,
        "range": 100,
        "actions": actions,
    }


def live_playback_with_prediction(
    video_path: Path,
    pose_model_path: Path,
    model: "FunscriptTCN",
    device: torch.device,
    vr_mode: bool = True,
    sbs_crop: str = "left",
    start_time: float | None = None,
    duration: float | None = None,
    target_fps: float = 30.0,
    frame_size: int = 640,
    max_persons: int = 10,
    seq_len: int = 120,
    emb_mean: np.ndarray | None = None,
    emb_std: np.ndarray | None = None,
    flow_mean: np.ndarray | None = None,
    flow_std: np.ndarray | None = None,
    multiclass: bool = False,
    max_partners: int = 5,
    max_beholders: int = 1,
) -> np.ndarray:
    """Live video playback with simultaneous feature extraction and TCN prediction.

    Runs the full decode → YOLO → RAFT → TCN pipeline on a background thread
    while displaying the video in real time via a tkinter GUI. Predictions are
    overlaid as they become available.

    Returns the final predictions array (for downstream funscript saving).
    """
    import os
    import signal
    import threading
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from PIL import Image, ImageTk

    #MAX_FRAME_BUF = 600       # max decoded frames to keep in RAM (~150MB at 640px)
    MAX_FRAME_BUF = 200
    STATS_EVERY_N = 8         # update stats panel every N display ticks
    GRAPH_EVERY_N = 4         # update matplotlib graph every N display ticks

    # ── Shared state (protected by _lock) ─────────────────────────────────
    _lock = threading.Lock()
    _state: dict = {
        "frames":           {},     # frame_idx -> np.ndarray RGB [H,W,3]
        "predictions":      None,   # np.ndarray or None (replaced after each chunk)
        "decode_fps":       0.0,
        "yolo_fps":         0.0,
        "frames_decoded":   0,
        "frames_predicted": 0,
        "total_est":        0,
        "status":           "Initializing…",
        "done":             False,
        "error":            None,
    }

    # ── Worker thread ──────────────────────────────────────────────────────
    def _worker() -> None:
        try:
            with _lock:
                _state["status"] = "Loading pose model…"

            pose_model = load_pose_model(
                model_name="yolo11m-pose",
                model_path=str(pose_model_path),
                device=str(device),
            )
            extractor = SinglePassExtractor(
                pose_model, max_persons=max_persons, n_keypoints=21,
                confidence_threshold=0.02, device=str(device),
                multiclass=multiclass,
                max_partners=max_partners,
                max_beholders=max_beholders,
            )

            # Estimate total frames
            total_est = 0
            try:
                from torchcodec.decoders import VideoDecoder as _VD
                _tmp = _VD(str(video_path))
                src_fps = _tmp.metadata.average_fps or 30.0
                src_n   = _tmp.metadata.num_frames or 0
                _tmp = None
                eff = target_fps or src_fps
                if duration:
                    total_est = int(duration * eff)
                elif src_fps > 0:
                    total_est = int(src_n * eff / src_fps)
            except Exception:
                pass
            with _lock:
                _state["total_est"] = total_est
                _state["status"]    = "Processing video…"

            crop_left = vr_mode and sbs_crop == "left"
            batch_size = 32
            flow_size  = 320

            all_kp:   list[np.ndarray] = []
            all_emb:  list[np.ndarray] = []
            all_flow: list[np.ndarray] = []
            prev_flow_frame: np.ndarray | None = None
            frame_count = 0
            t0_total  = time.perf_counter()
            t_yolo    = 0.0
            n_yolo    = 0

            for chunk_np in stream_video_gpu(
                video_path,
                device=str(device),
                crop_left_half=crop_left,
                target_size=frame_size,
                target_fps=target_fps,
                start_time=start_time,
                duration=duration,
                chunk_size=256,
                as_numpy=True,
            ):
                if not _running[0]:
                    break
                chunk_len = len(chunk_np)

                strategy = "suspend"
                if strategy == "evict":
                    # Store decoded frames (evict oldest when buffer full)
                    with _lock:
                        for i, f in enumerate(chunk_np):
                            _state["frames"][frame_count + i] = f
                        while len(_state["frames"]) > MAX_FRAME_BUF:
                            oldest = min(_state["frames"])
                            del _state["frames"][oldest]
                elif strategy == "suspend":
                    # Wait until buffer has room, then store all frames
                    while _running[0]:
                        with _lock:
                            buf_sz = len(_state["frames"])
                        if buf_sz + chunk_len <= MAX_FRAME_BUF:
                            break
                        time.sleep(0.05)
                    if not _running[0]:
                        break
                    with _lock:
                        for i, f in enumerate(chunk_np):
                            _state["frames"][frame_count + i] = f


                pose_frames = list(chunk_np)
                flow_frames = [cv2.resize(f, (flow_size, flow_size)) for f in pose_frames]

                # YOLO single-pass
                t1 = time.perf_counter()
                kp, emb = extract_single_pass_batched(extractor, pose_frames, batch_size)
                t_yolo += time.perf_counter() - t1
                n_yolo += chunk_len
                all_kp.append(kp)
                all_emb.append(emb)
                del pose_frames

                # RAFT optical flow
                if prev_flow_frame is not None:
                    fi = np.stack([prev_flow_frame] + flow_frames)
                    fc = compute_flow_raft_batched(
                        fi, output_features=64, device=str(device), batch_size=64,
                    )
                    all_flow.append(fc[1:])
                else:
                    fi = np.stack(flow_frames)
                    fc = compute_flow_raft_batched(
                        fi, output_features=64, device=str(device), batch_size=64,
                    )
                    all_flow.append(fc)
                prev_flow_frame = flow_frames[-1]
                del flow_frames

                frame_count += chunk_len
                elapsed = time.perf_counter() - t0_total
                decode_fps = frame_count / elapsed if elapsed > 0 else 0.0
                yolo_fps   = n_yolo / t_yolo if t_yolo > 0 else 0.0

                # Run TCN on all accumulated features so far
                preds_new = None
                if frame_count >= seq_len:
                    kp_a  = np.concatenate(all_kp)
                    emb_a = np.concatenate(all_emb)
                    fl_a  = np.concatenate(all_flow)
                    n = min(len(kp_a), len(emb_a), len(fl_a))
                    kp_a = kp_a[:n]; emb_a = emb_a[:n]; fl_a = fl_a[:n]
                    if emb_mean is not None:
                        emb_a = (emb_a - emb_mean) / (emb_std + 1e-8)
                    if flow_mean is not None:
                        fl_a = (fl_a - flow_mean) / (flow_std + 1e-8)
                    preds_new = sliding_window_predict(
                        model, kp_a, emb_a, fl_a, device, seq_len, stride=60,
                    )

                with _lock:
                    _state["decode_fps"]       = decode_fps
                    _state["yolo_fps"]         = yolo_fps
                    _state["frames_decoded"]   = frame_count
                    if preds_new is not None:
                        _state["predictions"]      = preds_new
                        _state["frames_predicted"] = len(preds_new)

            # Final prediction pass on complete data
            kp_a  = np.concatenate(all_kp)
            emb_a = np.concatenate(all_emb)
            fl_a  = np.concatenate(all_flow)
            n = min(len(kp_a), len(emb_a), len(fl_a))
            kp_a = kp_a[:n]; emb_a = emb_a[:n]; fl_a = fl_a[:n]
            if emb_mean is not None:
                emb_a = (emb_a - emb_mean) / (emb_std + 1e-8)
            if flow_mean is not None:
                fl_a = (fl_a - flow_mean) / (flow_std + 1e-8)
            preds_final = sliding_window_predict(
                model, kp_a, emb_a, fl_a, device, seq_len, stride=60,
            )

            extractor.close()
            with _lock:
                _state["predictions"]      = preds_final
                _state["frames_predicted"] = len(preds_final)
                _state["status"]           = "Done"
                _state["done"]             = True

        except Exception as exc:
            import traceback
            traceback.print_exc()
            with _lock:
                _state["error"]  = exc
                _state["status"] = f"Error: {exc}"
                _state["done"]   = True

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    # ── Helper: draw position meter (adapted from visualize_data.py) ───────
    def _draw_position_meter(frame_bgr: np.ndarray, position: float) -> np.ndarray:
        frame = frame_bgr.copy()
        h, w = frame.shape[:2]
        pos = float(np.clip(position, 0.0, 1.0))
        mw  = max(14, w // 45)
        pad = max(6, w // 100)
        x1, x2 = w - mw - pad, w - pad
        y1, y2  = pad, h - pad
        th = max(1, y2 - y1)
        lc = max(42, w // 14)
        sx = max(0, x1 - lc)
        roi = frame[y1:y2, sx:w].copy()
        frame[y1:y2, sx:w] = cv2.addWeighted(roi, 0.38, np.zeros_like(roi), 0.62, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 55, 55), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)
        fh = int(pos * th)
        if fh > 0:
            fy = y2 - fh
            r  = int(80 + 170 * pos)
            g  = int(130 - 30 * pos)
            b  = int(240 - 160 * pos)
            cv2.rectangle(frame, (x1, fy), (x2, y2), (b, g, r), -1)
        iy = max(y1, min(y2, y2 - int(pos * th)))
        cv2.rectangle(frame, (x1 - 3, iy - 2), (x2 + 3, iy + 2), (255, 255, 255), -1)
        lbl = f"{pos:.2f}"
        fs  = max(0.38, w / 1600.0)
        (tw, tht), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(frame, lbl, (x1 - tw - 6, max(y1 + tht, min(y2 - 2, iy + tht // 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        for val in (1.0, 0.5, 0.0):
            vy = max(y1 + 8, min(y2, y2 - int(val * th) + 4))
            cv2.putText(frame, f"{val:.1f}", (x1 + 2, vy),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.28, w / 2200.0),
                        (130, 130, 130), 1, cv2.LINE_AA)
        return frame

    # ── Build tkinter GUI ──────────────────────────────────────────────────
    root = tk.Tk()
    root.title("VideoToMotion — Live Prediction")
    root.configure(bg="#1e1e1e")
    root.resizable(True, True)

    _running = [True]
    _tk_photo: list = [None]         # keep reference to avoid GC

    # Playback state
    _pos      = [0]                  # current display frame index
    _playing  = [False]              # start paused until buffer fills
    _speed    = [1.0]
    _t_start  = [time.perf_counter()]
    _f_start  = [0]                  # frame index when timer was last reset
    _buf_wait = [True]               # True while waiting for buffer to fill
    _tick     = [0]                  # display update counter

    # Buffer thresholds
    BUF_START_THRESHOLD = MAX_FRAME_BUF       # must fill completely before initial play
    BUF_RESUME_THRESHOLD = MAX_FRAME_BUF
    BUF_LOW_THRESHOLD    = 120                 # pause playback when buffer drops below this

    frame_delay_ms = max(16, int(1000.0 / target_fps))

    # ── Layout ────────────────────────────────────────────────────────────
    # Row 1: video canvas  +  stats panel
    top_row = tk.Frame(root, bg="#1e1e1e")
    top_row.pack(fill=tk.BOTH, expand=False, padx=4, pady=(4, 0))

    video_canvas = tk.Canvas(
        top_row, bg="#000000", width=frame_size, height=frame_size,
        highlightthickness=0,
    )
    video_canvas.pack(side=tk.LEFT, padx=(0, 4))

    # Stats panel
    stats_outer = tk.Frame(top_row, bg="#252526", width=230)
    stats_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
    stats_outer.pack_propagate(False)

    tk.Label(
        stats_outer, text="Live Stats", bg="#252526", fg="#cccccc",
        font=("Segoe UI", 10, "bold"), anchor="w",
    ).pack(fill=tk.X, padx=8, pady=(8, 4))

    stats_text = tk.Text(
        stats_outer, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9),
        borderwidth=0, highlightthickness=0, state=tk.DISABLED,
        wrap=tk.WORD, height=22, width=26,
    )
    stats_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    stats_text.tag_configure("head",  foreground="#4ec9b0", font=("Consolas", 9, "bold"))
    stats_text.tag_configure("val",   foreground="#ce9178")
    stats_text.tag_configure("ok",    foreground="#6a9955")
    stats_text.tag_configure("warn",  foreground="#d79921")
    stats_text.tag_configure("err",   foreground="#f44747")
    stats_text.tag_configure("dim",   foreground="#555555")

    # Row 2: matplotlib prediction timeline
    graph_frame = tk.Frame(root, bg="#1e1e1e")
    graph_frame.pack(fill=tk.X, padx=4, pady=(4, 0))

    fig = plt.figure(figsize=(10, 2.0), facecolor="#1e1e1e")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#252526")
    ax.tick_params(colors="#888888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Position", color="#888888", fontsize=8)
    ax.set_xlabel("Time (s)",  color="#888888", fontsize=8)
    for g_val in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.axhline(g_val, color="#333333", lw=0.5, linestyle=":")
    [pred_line]   = ax.plot([], [], lw=1.5,  color="#e37933", label="Prediction", animated=True)
    [cursor_line] = ax.plot([], [], lw=1.5,  color="#4ee344", linestyle="--", label="Now", animated=True)
    ax.legend(fontsize=7, labelcolor="#888888", facecolor="#252526", edgecolor="#444444",
              loc="upper left")
    fig.tight_layout(pad=0.6)

    timeline_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    timeline_canvas.get_tk_widget().pack(fill=tk.X)

    # Initial draw to capture background for blitting
    timeline_canvas.draw()
    _graph_bg = [timeline_canvas.copy_from_bbox(ax.bbox)]

    # Make timeline clickable for seeking
    def _on_timeline_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        target_frame = max(0, int(event.xdata * target_fps))
        _pos[0] = target_frame
        _t_start[0] = time.perf_counter()
        _f_start[0] = target_frame
        _buf_wait[0] = False  # user manually seeked

    timeline_canvas.mpl_connect("button_press_event", _on_timeline_click)

    # Row 2.5: seek slider spanning full video duration
    _seek_var = tk.IntVar(value=0)
    _seek_updating = [False]  # prevent feedback loops

    def _on_seek_slider(val: str) -> None:
        if _seek_updating[0]:
            return
        frame = int(float(val))
        _pos[0] = frame
        _t_start[0] = time.perf_counter()
        _f_start[0] = frame
        _buf_wait[0] = False

    seek_frame = tk.Frame(root, bg="#1e1e1e")
    seek_frame.pack(fill=tk.X, padx=4, pady=(2, 0))
    seek_slider = tk.Scale(
        seek_frame, from_=0, to=1, orient=tk.HORIZONTAL,
        variable=_seek_var, command=_on_seek_slider,
        bg="#1e1e1e", fg="#888888", troughcolor="#3c3c3c",
        highlightthickness=0, showvalue=False, length=600,
        sliderrelief=tk.FLAT, sliderlength=12,
    )
    seek_slider.pack(fill=tk.X, expand=True)
    seek_time_label = tk.Label(seek_frame, text="0:00 / 0:00",
                               bg="#1e1e1e", fg="#888888", font=("Consolas", 8))
    seek_time_label.pack()

    # Row 3: controls
    ctrl_frame = tk.Frame(root, bg="#1e1e1e")
    ctrl_frame.pack(fill=tk.X, padx=4, pady=(4, 4))

    _BTN = dict(bg="#3c3c3c", fg="#cccccc", activebackground="#505050",
                activeforeground="#ffffff", relief=tk.FLAT,
                font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2")

    def _toggle_play() -> None:
        _playing[0] = not _playing[0]
        if _playing[0]:
            _t_start[0] = time.perf_counter()
            _f_start[0] = _pos[0]
            btn_play.config(text="⏸ Pause")
        else:
            btn_play.config(text="▶ Play")

    def _seek(delta: int) -> None:
        _pos[0] = max(0, _pos[0] + delta)
        _t_start[0] = time.perf_counter()
        _f_start[0] = _pos[0]
        _buf_wait[0] = False  # user manually seeked, start playing immediately

    tk.Button(ctrl_frame, text="◀◀", command=lambda: _seek(-int(target_fps * 5)), **_BTN).pack(side=tk.LEFT, padx=2)
    tk.Button(ctrl_frame, text="◀",  command=lambda: _seek(-int(target_fps)),      **_BTN).pack(side=tk.LEFT, padx=2)
    btn_play = tk.Button(ctrl_frame, text="▶ Play", command=_toggle_play, **_BTN)
    btn_play.pack(side=tk.LEFT, padx=2)
    tk.Button(ctrl_frame, text="▶",  command=lambda: _seek(int(target_fps)),       **_BTN).pack(side=tk.LEFT, padx=2)
    tk.Button(ctrl_frame, text="▶▶", command=lambda: _seek(int(target_fps * 5)),   **_BTN).pack(side=tk.LEFT, padx=2)

    frame_label = tk.Label(ctrl_frame, text="Frame: 0  |  0.0s",
                           bg="#1e1e1e", fg="#888888", font=("Consolas", 9))
    frame_label.pack(side=tk.LEFT, padx=12)

    pos_label = tk.Label(ctrl_frame, text="pos=…",
                         bg="#1e1e1e", fg="#4ec9b0", font=("Consolas", 10, "bold"))
    pos_label.pack(side=tk.LEFT, padx=4)

    speed_var = tk.DoubleVar(value=1.0)
    tk.Label(ctrl_frame, text="Speed:", bg="#1e1e1e", fg="#888888",
             font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
    for spd in (0.25, 0.5, 1.0, 2.0):
        def _set_spd(s: float = spd) -> None:
            _speed[0] = s
            speed_var.set(s)
            _t_start[0] = time.perf_counter()
            _f_start[0] = _pos[0]
        tk.Radiobutton(
            ctrl_frame, text=f"{spd}×", variable=speed_var, value=spd,
            bg="#1e1e1e", fg="#cccccc", selectcolor="#1e1e1e",
            activebackground="#1e1e1e", font=("Segoe UI", 9),
            command=_set_spd,
        ).pack(side=tk.LEFT, padx=2)

    worker_status_label = tk.Label(ctrl_frame, text="● Starting…",
                                   bg="#1e1e1e", fg="#d79921", font=("Segoe UI", 9))
    worker_status_label.pack(side=tk.RIGHT, padx=8)

    root.bind("<space>",   lambda e: _toggle_play())
    root.bind("<Left>",    lambda e: _seek(-int(target_fps)))
    root.bind("<Right>",   lambda e: _seek(int(target_fps)))
    # All close paths go through _on_close so pending after callbacks are cancelled
    _after_id = [None]  # track the scheduled tick so we can cancel it

    def _on_close() -> None:
        _running[0] = False
        if _after_id[0] is not None:
            try:
                root.after_cancel(_after_id[0])
            except Exception:
                pass
            _after_id[0] = None
        # Close matplotlib BEFORE destroying Tk root — avoids hang in plt.close
        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.bind("<q>",      lambda e: _on_close())
    root.bind("<Escape>", lambda e: _on_close())

    # Ctrl+C handler — schedule close on the main thread
    def _sigint_handler(sig, frame) -> None:
        _running[0] = False
        try:
            root.after(0, _on_close)
        except Exception:
            os._exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)
    # ── Update helpers ─────────────────────────────────────────────────────
    def _update_stats(st: dict, cur_frame: int) -> None:
        n_dec  = st["frames_decoded"]
        n_pred = st["frames_predicted"]
        t_cur  = cur_frame / max(target_fps, 1.0)
        lag    = max(0.0, (n_dec - n_pred) / max(target_fps, 1.0))
        buf_sz = len(st["frames"])
        status = st["status"]
        total_est = st["total_est"]

        stats_text.config(state=tk.NORMAL)
        stats_text.delete("1.0", tk.END)

        def row(label: str, value: str, tag: str = "val") -> None:
            stats_text.insert(tk.END, f"  {label:<18}", "dim")
            stats_text.insert(tk.END, f"{value}\n", tag)

        stats_text.insert(tk.END, "  PLAYBACK\n", "head")
        row("Position:", f"{t_cur:.1f}s")
        row("Frame:", f"{cur_frame}")
        row("Speed:", f"{_speed[0]}×")
        row("Playing:", "Yes" if _playing[0] else "No", "ok" if _playing[0] else "warn")
        if _buf_wait[0]:
            stats_text.insert(tk.END, "  ⏸ Buffering…\n", "warn")

        stats_text.insert(tk.END, "\n  PIPELINE\n", "head")
        row("Decode fps:",  f"{st['decode_fps']:.1f}")
        row("YOLO fps:",    f"{st['yolo_fps']:.1f}")
        row("Decoded:",     f"{n_dec} fr")
        row("Predicted:",   f"{n_pred} fr")
        if total_est > 0:
            pct = n_dec / total_est * 100
            row("Progress:", f"{pct:.0f}%  ({n_dec}/{total_est})")

        lag_tag = "ok" if lag < 5 else ("warn" if lag < 15 else "err")
        row("Inf. lag:",  f"{lag:.1f}s", lag_tag)
        buf_tag = "ok" if buf_sz > 60 else ("warn" if buf_sz > 10 else "err")
        row("Frame buf:", f"{buf_sz} fr", buf_tag)

        stats_text.insert(tk.END, "\n  STATUS\n", "head")
        s_tag = "ok" if status == "Done" else ("err" if "Error" in status else "warn")
        stats_text.insert(tk.END, f"  {status}\n", s_tag)
        stats_text.config(state=tk.DISABLED)

        wc = "● Done" if status == "Done" else ("● Error" if "Error" in status else "● Processing")
        wfg = "#6a9955" if status == "Done" else ("#f44747" if "Error" in status else "#d79921")
        worker_status_label.config(text=wc, fg=wfg)

    _prev_xlim = [0.0, 30.0]  # track previous x-axis limits for blitting

    def _update_graph(preds: np.ndarray, cur_frame: int) -> None:
        if preds is None or len(preds) == 0:
            return
        t_all = np.arange(len(preds)) / target_fps
        t_cur = cur_frame / target_fps
        half  = 15.0  # half-window in seconds
        t0w   = max(0.0, t_cur - half)
        t1w   = t0w + half * 2
        if t1w > t_all[-1]:
            t1w = t_all[-1]
            t0w = max(0.0, t1w - half * 2)
        mask  = (t_all >= t0w) & (t_all <= t1w)
        pred_line.set_data(t_all[mask], preds[mask])
        cursor_line.set_data([t_cur, t_cur], [-0.05, 1.05])

        # If x-limits changed, we need a full redraw to update axes/ticks
        new_xlim = [t0w, t1w]
        if abs(new_xlim[0] - _prev_xlim[0]) > 0.5 or abs(new_xlim[1] - _prev_xlim[1]) > 0.5:
            ax.set_xlim(t0w, t1w)
            _prev_xlim[:] = new_xlim
            timeline_canvas.draw()
            _graph_bg[0] = timeline_canvas.copy_from_bbox(ax.bbox)

        # Blit: restore background, draw animated artists, blit
        timeline_canvas.restore_region(_graph_bg[0])
        ax.draw_artist(pred_line)
        ax.draw_artist(cursor_line)
        timeline_canvas.blit(ax.bbox)

    # ── Main display loop ──────────────────────────────────────────────────
    def _tick_display() -> None:
        if not _running[0]:
            return
        try:
            _tick_display_inner()
        except Exception:
            pass  # silently stop if widgets were destroyed during close

    def _tick_display_inner() -> None:
        # Snapshot shared state (minimal lock hold)
        with _lock:
            st     = dict(_state)            # shallow copy of scalars/refs
            frames = dict(_state["frames"])  # copy frame dict
            preds  = _state["predictions"]   # np.ndarray ref (immutable content)

        n_decoded = st["frames_decoded"]
        n_pred    = len(preds) if preds is not None else 0

        # Advance playback clock
        if _playing[0]:
            elapsed_s  = (time.perf_counter() - _t_start[0]) * _speed[0]
            target_idx = int(_f_start[0] + elapsed_s * target_fps)
        else:
            target_idx = _pos[0]

        # Buffer management: auto-pause when buffer is low, auto-resume when refilled
        buf_sz = len(frames)
        worker_done = st["done"]

        if _buf_wait[0]:
            # Decide which threshold to use: full buffer on initial load,
            # half-buffer when resuming after a catch-up stall
            needed = BUF_START_THRESHOLD if _pos[0] == 0 else BUF_RESUME_THRESHOLD
            if buf_sz >= needed or worker_done:
                # Buffer is full enough — start/resume playback
                _buf_wait[0] = False
                if not _playing[0]:
                    _playing[0] = True
                    _t_start[0] = time.perf_counter()
                    _f_start[0] = _pos[0]
                    btn_play.config(text="⏸ Pause")
                    # Recalculate target after resuming
                    elapsed_s  = (time.perf_counter() - _t_start[0]) * _speed[0]
                    target_idx = int(_f_start[0] + elapsed_s * target_fps)
        elif not worker_done:
            # Check if we've caught up and the buffer is running low
            if buf_sz < BUF_LOW_THRESHOLD or (frames and target_idx > max(frames.keys())):
                _buf_wait[0] = True
                _playing[0] = False
                btn_play.config(text="▶ Play")
                # Freeze clock so position doesn't jump when we resume
                _t_start[0] = time.perf_counter()
                _f_start[0] = _pos[0]
                target_idx = _pos[0]

        _pos[0] = target_idx

        # Auto-stop when worker done and all frames displayed
        if st["done"] and n_pred > 0 and _pos[0] >= n_pred - 1:
            _playing[0] = False
            btn_play.config(text="▶ Play")

        # ── Render frame ──────────────────────────────────────────────────
        frame_rgb = frames.get(_pos[0])
        if frame_rgb is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw position meter if we have a prediction
            pred_val: float | None = None
            if preds is not None and len(preds) > 0:
                idx = min(_pos[0], len(preds) - 1)
                pred_val = float(preds[idx])
            if pred_val is not None:
                frame_bgr = _draw_position_meter(frame_bgr, pred_val)
                pos_label.config(text=f"pos={pred_val:.3f}")
            else:
                pos_label.config(text="pos=…")

            # Buffering banner
            if _buf_wait[0]:
                cv2.putText(frame_bgr, "BUFFERING…", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=img)
            _tk_photo[0] = photo
            h_f, w_f = frame_bgr.shape[:2]
            video_canvas.config(width=w_f, height=h_f)
            video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Evict frames behind the playback position to free memory
        if _playing[0] and _pos[0] > 30:
            evict_below = _pos[0] - 30  # keep a small lookback for seeking
            with _lock:
                to_del = [k for k in _state["frames"] if k < evict_below]
                for k in to_del:
                    del _state["frames"][k]

        # Frame / time label
        t_sec = _pos[0] / max(target_fps, 1.0)
        frame_label.config(text=f"Frame: {_pos[0]} / {n_decoded}  |  {t_sec:.1f}s")

        # Update seek slider range and position
        max_frame = max(n_decoded, n_pred, 1)
        if seek_slider.cget("to") != max_frame:
            seek_slider.config(to=max_frame)
        _seek_updating[0] = True
        _seek_var.set(_pos[0])
        _seek_updating[0] = False
        total_sec = max_frame / max(target_fps, 1.0)
        cur_min, cur_s = divmod(int(t_sec), 60)
        tot_min, tot_s = divmod(int(total_sec), 60)
        seek_time_label.config(text=f"{cur_min}:{cur_s:02d} / {tot_min}:{tot_s:02d}")

        # Periodic stats + graph updates
        _tick[0] += 1
        if _tick[0] % STATS_EVERY_N == 0:
            _update_stats(st, _pos[0])
        if _tick[0] % GRAPH_EVERY_N == 0:
            _update_graph(preds, _pos[0])

        try:
            _after_id[0] = root.after(frame_delay_ms, _tick_display)
        except Exception:
            pass

    # Kick off the display loop once the window is ready
    _after_id[0] = root.after(100, _tick_display)
    root.mainloop()

    # ── Cleanup ────────────────────────────────────────────────────────────
    _running[0] = False
    # plt.close already called in _on_close; nothing matplotlib to clean up here

    # Wait for worker thread to finish (timeout to avoid hanging forever)
    worker_thread.join(timeout=5.0)

    with _lock:
        final_preds = _state["predictions"]
        err         = _state["error"]

    if err is not None:
        print(f"Warning: worker thread encountered an error: {err}")

    result = final_preds if final_preds is not None else np.array([], dtype=np.float32)

    # Force exit if threads are still alive (e.g. RAFT/YOLO blocking)
    if worker_thread.is_alive():
        print("Worker thread still running — forcing exit.")
        os._exit(0)

    return result


def _plot_predictions(
    predictions: np.ndarray,
    labels: np.ndarray | None,
    fps: float,
    source_name: str,
    args,
) -> None:
    """Show / save a matplotlib prediction plot."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg" if args.plot else "Agg")
    time_axis = np.arange(len(predictions)) / fps
    fig, axes = plt.subplots(
        2 if labels is not None else 1, 1,
        figsize=(16, 6 if labels is not None else 4),
        sharex=True,
    )
    if labels is None:
        axes = [axes]

    axes[0].plot(time_axis, predictions, lw=1.5, color="darkorange", label="TCN prediction")
    if labels is not None:
        axes[0].plot(time_axis, labels, lw=1.2, alpha=0.7, color="steelblue", label="Ground truth")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Position")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"TCN Prediction — {source_name}")
    axes[0].grid(alpha=0.3)

    if labels is not None:
        error = predictions - labels
        axes[1].fill_between(time_axis, error, color="red", alpha=0.4, label="Error (pred - target)")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].set_ylabel("Error")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if args.save_plot:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_plot, dpi=150)
        print(f"Saved plot: {args.save_plot}")

    if args.plot:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN funscript prediction")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--scene", help="Scene ID, e.g. scene_00018_t00926_40s")
    input_group.add_argument("--video", type=Path, help="Path to mp4 video file")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/models/checkpoints_tcn/best_tcn.pt"))
    parser.add_argument("--pose-model", type=Path,
                        default=Path("data/models/pose/pose-vrlens-finetunes-large.pt"),
                        help="Path to YOLO pose model weights")
    parser.add_argument("--vr", dest="vr_mode", action="store_true", default=True,
                        help="Video is VR side-by-side (default)")
    parser.add_argument("--no-vr", dest="vr_mode", action="store_false",
                        help="Video is flat (non-VR)")
    parser.add_argument("--sbs-crop", default="left", choices=["left", "right"],
                        help="Which eye to use for VR SBS")
    parser.add_argument("--frame-size", type=int, default=640)
    parser.add_argument("--max-persons", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-time", type=float, default=None,
                        help="Start time in seconds for video extraction")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds to extract")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output funscript JSON path")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib plot of prediction vs ground truth")
    parser.add_argument("--save-plot", type=Path, default=None,
                        help="Save the plot to this path (PNG)")
    parser.add_argument("--playback", action="store_true",
                        help="Real-time video playback with prediction overlay")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    labels = None
    source_name = ""

    # ── Live playback path (--playback --video) ───────────────────────────
    if args.playback and args.video is not None:
        if not args.video.exists():
            print(f"Error: video file not found: {args.video}")
            sys.exit(1)

        model, model_cfg = load_model(args.checkpoint, device)
        is_multiclass = model_cfg.get("n_partners") is not None
        n_total = (model_cfg.get("n_partners", 5) + model_cfg.get("n_beholders", 1)) if is_multiclass else model_cfg.get("n_persons", 10)

        # Auto-select pose model for multiclass
        pose_model_path = args.pose_model
        if is_multiclass and "multiclass" not in str(pose_model_path):
            pose_model_path = Path("data/models/pose/vrlens-finetunes-multiclass-v2-yolo11m-pose.pt")
            print(f"  Auto-selected multiclass pose model: {pose_model_path}")

        emb_mean, emb_std, flow_mean, flow_std = load_stats(args.data_dir, n_persons=n_total)
        source_name = args.video.stem

        print(f"Starting live playback + prediction for {args.video}...")
        predictions = live_playback_with_prediction(
            args.video, pose_model_path, model, device,
            vr_mode=args.vr_mode, sbs_crop=args.sbs_crop,
            start_time=args.start_time, duration=args.duration,
            target_fps=args.fps, frame_size=args.frame_size,
            max_persons=n_total, seq_len=args.seq_len,
            emb_mean=emb_mean, emb_std=emb_std,
            flow_mean=flow_mean, flow_std=flow_std,
            multiclass=is_multiclass,
            max_partners=model_cfg.get("n_partners", 5),
            max_beholders=model_cfg.get("n_beholders", 1),
        )

        if len(predictions) == 0:
            print("No predictions produced (window closed before processing completed).")
            return

        n_frames = len(predictions)
        print(f"  Predictions: {n_frames} frames, "
              f"range=[{predictions.min():.4f}, {predictions.max():.4f}]")

        # Save funscript
        out_path = args.out or args.video.with_suffix(".funscript")
        funscript = predictions_to_funscript(predictions, args.fps, args.start_time or 0.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(funscript, f)
        print(f"Saved funscript: {out_path} ({len(funscript['actions'])} actions)")

        # Optional plot after playback
        if args.plot or args.save_plot:
            _plot_predictions(predictions, None, args.fps, source_name, args)
        return

    # ── Batch path (no --playback) ───────────────────────────────────────
    # Load model first to determine multiclass mode
    model, model_cfg = load_model(args.checkpoint, device)
    is_multiclass = model_cfg.get("n_partners") is not None
    n_total = (model_cfg.get("n_partners", 5) + model_cfg.get("n_beholders", 1)) if is_multiclass else model_cfg.get("n_persons", 10)

    if args.scene:
        # Load pre-extracted features from scene directory
        scene_dir = args.data_dir / "processed" / args.scene
        if not scene_dir.exists():
            print(f"Error: scene directory not found: {scene_dir}")
            sys.exit(1)

        kp_file = KP_FILE_MULTICLASS if is_multiclass else KP_FILE
        emb_file = EMB_FILE_MULTICLASS if is_multiclass else EMB_FILE

        missing = []
        for fname in [kp_file, emb_file, FLOW_FILE]:
            if not (scene_dir / fname).exists():
                missing.append(fname)
        if missing:
            print(f"Error: missing feature files in {scene_dir}:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)

        print(f"Loading features from {scene_dir}...")
        keypoints  = np.load(str(scene_dir / kp_file))
        embeddings = np.load(str(scene_dir / emb_file))
        flow       = np.load(str(scene_dir / FLOW_FILE))

        labels_path = scene_dir / "labels.npy"
        if labels_path.exists():
            labels = np.load(str(labels_path))

        source_name = args.scene

    else:
        # Extract features from video file
        if not args.video.exists():
            print(f"Error: video file not found: {args.video}")
            sys.exit(1)

        # Auto-select pose model for multiclass
        pose_model_path = args.pose_model
        if is_multiclass and "multiclass" not in str(pose_model_path):
            pose_model_path = Path("data/models/pose/vrlens-finetunes-multiclass-v2-yolo11m-pose.pt")
            print(f"  Auto-selected multiclass pose model: {pose_model_path}")

        print(f"Extracting features from {args.video}...")
        keypoints, embeddings, flow = extract_features_from_video(
            args.video, pose_model_path,
            vr_mode=args.vr_mode, sbs_crop=args.sbs_crop,
            frame_size=args.frame_size, max_persons=n_total,
            device=device,
            start_time=args.start_time,
            duration=args.duration,
            target_fps=args.fps,
            multiclass=is_multiclass,
            max_partners=model_cfg.get("n_partners", 5),
            max_beholders=model_cfg.get("n_beholders", 1),
        )

        source_name = args.video.stem

    n_frames = len(keypoints)
    print(f"  keypoints:  {keypoints.shape}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  flow:       {flow.shape}")
    print(f"  frames:     {n_frames} ({n_frames / args.fps:.1f}s @ {args.fps}fps)")

    # Normalize features
    emb_mean, emb_std, flow_mean, flow_std = load_stats(args.data_dir, n_persons=n_total)
    if emb_mean is not None:
        embeddings = (embeddings - emb_mean) / (emb_std + 1e-8)
        print("  Embeddings normalized (z-score)")
    if flow_mean is not None:
        flow = (flow - flow_mean) / (flow_std + 1e-8)
        print("  Flow normalized (z-score)")

    # Model already loaded above

    # Predict
    print(f"Predicting with sliding window (seq_len={args.seq_len}, stride={args.stride})...")
    predictions = sliding_window_predict(
        model, keypoints, embeddings, flow, device, args.seq_len, args.stride
    )
    print(f"  Output: {predictions.shape}, range=[{predictions.min():.4f}, {predictions.max():.4f}], "
          f"mean={predictions.mean():.4f}, std={predictions.std():.4f}")

    # Compare to ground truth labels if available
    if labels is not None:
        mse = float(np.mean((predictions - labels) ** 2))
        mae = float(np.mean(np.abs(predictions - labels)))
        print(f"  vs ground truth: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={mse**0.5:.4f}")

    # Save funscript
    if args.out is not None:
        funscript = predictions_to_funscript(predictions, args.fps, args.start_time or 0.0)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(funscript, f)
        print(f"Saved funscript: {args.out} ({len(funscript['actions'])} actions)")
    elif args.video is not None:
        # Auto-save funscript next to the video
        auto_out = args.video.with_suffix(".funscript")
        funscript = predictions_to_funscript(predictions, args.fps, args.start_time or 0.0)
        with open(auto_out, "w") as f:
            json.dump(funscript, f)
        print(f"Saved funscript: {auto_out} ({len(funscript['actions'])} actions)")

    # Plot
    if args.plot or args.save_plot:
        _plot_predictions(predictions, labels, args.fps, source_name, args)


if __name__ == "__main__":
    main()
