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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.tcn import FunscriptTCN
from src.data.pose import load_pose_model, extract_pose_batch
from src.data.embeddings import YOLOEmbeddingExtractor, EMBED_DIM
from src.data.flow import compute_flow_raft_batched
from src.config import Config
from src.data.curation import (
    discover_scenes, flow_path,
    resolve_keypoints_path, resolve_flow_path,
)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

KP_FILE = "keypoints/pose-vrlens-finetunes-large.npy"
EMB_FILE = "embeddings/pose-vrlens-finetunes-large.npy"
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


def load_model(checkpoint_path: Path, device: torch.device) -> FunscriptTCN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = FunscriptTCN(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.6f}")
    return model


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
    chunk_size: int = 512,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract keypoints, embeddings, and flow features from a raw video.

    Streaming pipeline: decodes and processes one chunk of frames at a time so
    only ~chunk_size frame images are held in RAM simultaneously. Feature arrays
    (keypoints, embeddings, flow) are accumulated as they are produced and are
    much smaller than the raw frames.
    """
    timings: dict[str, float] = {"model_load": 0.0, "video_decode": 0.0,
                                  "yolo_pose": 0.0, "yolo_embed": 0.0, "raft_flow": 0.0}

    # ── Load pose model ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"Loading pose model from {pose_model_path}...")
    pose_model = load_pose_model(
        model_name="yolo11m-pose",
        model_path=str(pose_model_path),
        device=str(device),
    )
    extractor = YOLOEmbeddingExtractor(
        pose_model, layer_idx=10, max_persons=max_persons, device=str(device),
    )
    timings["model_load"] = time.perf_counter() - t0

    batch_size = 32
    flow_size = max(64, int(0.5 * 640))  # 320

    all_keypoints: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_flow: list[np.ndarray] = []

    # prev_flow_frame: last flow-size frame from previous chunk, for RAFT continuity
    prev_flow_frame: np.ndarray | None = None
    n_frames = 0
    first_chunk = True

    print(f"  Streaming video in chunks of {chunk_size} frames...")
    t_stream_start = time.perf_counter()

    totalframes = None
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0 and target_fps is not None:
            totalframes = int(totalframes * (target_fps / video_fps))
        cap.release()
    if duration is not None and target_fps is not None:
        totalframes = min(totalframes or float("inf"), int(duration * target_fps))
    tp = tqdm(total=(totalframes // chunk_size), unit=f"{chunk_size} frames", desc="Processing video", dynamic_ncols=True)

    for chunk in _stream_frames_ffmpeg(
        video_path,
        target_size=frame_size,
        crop_left_half=(vr_mode and sbs_crop == "left"),
        start_time=start_time,
        duration=duration,
        target_fps=target_fps,
        chunk_size=chunk_size,
    ):
        t_decode_end = time.perf_counter()
        timings["video_decode"] += t_decode_end - (t_stream_start if first_chunk else t_proc_end)

        chunk_n = len(chunk)
        n_frames += chunk_n

        # ── Detect SBS on first chunk only ───────────────────────────────
        if first_chunk:
            h, w = chunk[0].shape[:2]
            is_vr_sbs = w > h * 1.8
            decode_method = "ffmpeg+nvdec" if (h == frame_size and w == frame_size) else "cv2/cpu"
            #print(f"  First chunk: {chunk_n} frames {w}×{h} via {decode_method}")
            first_chunk = False

        # ── Preprocess: pose frames & flow frames ─────────────────────────
        h, w = chunk[0].shape[:2]
        if h == frame_size and w == frame_size:
            pose_frames = chunk
        elif vr_mode and is_vr_sbs:
            half_w = w // 2
            if sbs_crop == "left":
                pose_frames = [cv2.resize(f[:, :half_w], (frame_size, frame_size)) for f in chunk]
            else:
                pose_frames = [cv2.resize(f[:, half_w:], (frame_size, frame_size)) for f in chunk]
        else:
            pose_frames = [cv2.resize(f, (frame_size, frame_size)) for f in chunk]

        flow_frames = [cv2.resize(f, (flow_size, flow_size)) for f in chunk]
        del chunk  # free raw frames ASAP

        # ── YOLO pose ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        kpts = _extract_pose_batched(pose_model, pose_frames, max_persons, chunk_n, batch_size)
        all_keypoints.append(kpts)
        timings["yolo_pose"] += time.perf_counter() - t0

        # ── YOLO embed ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunk_emb = []
        for i in range(0, chunk_n, batch_size):
            batch = np.stack(pose_frames[i : i + batch_size])
            chunk_emb.append(extractor.extract_batch(batch))
        all_embeddings.append(np.concatenate(chunk_emb, axis=0))
        timings["yolo_embed"] += time.perf_counter() - t0

        del pose_frames

        # ── RAFT flow ─────────────────────────────────────────────────────
        # Prepend the last frame of the previous chunk so the first frame of
        # this chunk gets a valid flow vector instead of a copy of frame 1.
        t0 = time.perf_counter()
        if prev_flow_frame is not None:
            flow_input = np.stack([prev_flow_frame] + flow_frames)  # [chunk_n+1, H, W, C]
            flow_chunk = compute_flow_raft_batched(
                flow_input, output_features=64, device=str(device), batch_size=64,
            )
            # flow_input[0]=prev_last, flow_input[1]=frame0, ...
            # flow_chunk[1] = flow(prev_last→frame0) — correct for frame0
            # flow_chunk[2] = flow(frame0→frame1) — correct for frame1, etc.
            all_flow.append(flow_chunk[1:])  # [chunk_n, 64]
        else:
            flow_input = np.stack(flow_frames)
            flow_chunk = compute_flow_raft_batched(
                flow_input, output_features=64, device=str(device), batch_size=64,
            )
            all_flow.append(flow_chunk)
        prev_flow_frame = flow_frames[-1]
        del flow_frames
        timings["raft_flow"] += time.perf_counter() - t0

        t_proc_end = time.perf_counter()
        tp.update(1)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN funscript prediction")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--scene", help="Scene ID, e.g. scene_00018_t00926_40s")
    input_group.add_argument("--video", type=Path, help="Path to mp4 video file")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/models/checkpoints_tcn/best_tcn.pt"))
    parser.add_argument("--pose-model", type=Path,
                        default=Path("pose-vrlens-finetunes-large.pt"),
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
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load or extract features ────────────────────────────────────────
    labels = None

    if args.scene:
        # Load pre-extracted features from scene directory
        scene_dir = args.data_dir / "processed" / args.scene
        if not scene_dir.exists():
            print(f"Error: scene directory not found: {scene_dir}")
            sys.exit(1)

        missing = []
        for fname in [KP_FILE, EMB_FILE, FLOW_FILE]:
            if not (scene_dir / fname).exists():
                missing.append(fname)
        if missing:
            print(f"Error: missing feature files in {scene_dir}:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)

        print(f"Loading features from {scene_dir}...")
        keypoints  = np.load(str(scene_dir / KP_FILE))
        embeddings = np.load(str(scene_dir / EMB_FILE))
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

        print(f"Extracting features from {args.video}...")
        keypoints, embeddings, flow = extract_features_from_video(
            args.video, args.pose_model,
            vr_mode=args.vr_mode, sbs_crop=args.sbs_crop,
            frame_size=args.frame_size, max_persons=args.max_persons,
            device=device,
            start_time=args.start_time,
            duration=args.duration,
            target_fps=args.fps,
        )

        source_name = args.video.stem

    n_frames = len(keypoints)
    print(f"  keypoints:  {keypoints.shape}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  flow:       {flow.shape}")
    print(f"  frames:     {n_frames} ({n_frames / args.fps:.1f}s @ {args.fps}fps)")

    # Normalize features
    emb_mean, emb_std, flow_mean, flow_std = load_stats(args.data_dir)
    if emb_mean is not None:
        embeddings = (embeddings - emb_mean) / (emb_std + 1e-8)
        print("  Embeddings normalized (z-score)")
    if flow_mean is not None:
        flow = (flow - flow_mean) / (flow_std + 1e-8)
        print("  Flow normalized (z-score)")

    # Load model
    model = load_model(args.checkpoint, device)

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
        matplotlib.use("TkAgg" if args.plot else "Agg")
        time_axis = np.arange(n_frames) / args.fps

        fig, axes = plt.subplots(2 if labels is not None else 1, 1,
                                  figsize=(16, 6 if labels is not None else 4),
                                  sharex=True)
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


if __name__ == "__main__":
    main()
