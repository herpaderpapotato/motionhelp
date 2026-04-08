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
from src.data.pose import load_pose_model
from src.data.extraction import SinglePassExtractor, extract_single_pass_batched
from src.data.decode import stream_video_gpu
from src.data.flow import compute_flow_raft_batched

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


def playback_with_prediction(
    video_path: Path,
    predictions: np.ndarray,
    fps: float,
    vr_mode: bool,
    sbs_crop: str,
    start_time: float = 0.0,
    display_height: int = 720,
) -> None:
    """Real-time video playback with prediction graph overlay."""
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(video_path), device="cuda", dimension_order="NHWC")
        src_fps = decoder.metadata.average_fps or 30.0
    except Exception:
        print("Error: torchcodec CUDA required for playback mode")
        return

    start_frame = int(start_time * src_fps) if start_time else 0
    step = src_fps / fps if fps < src_fps else 1.0
    n_pred = len(predictions)
    graph_height = 140

    cv2.namedWindow("Prediction Playback", cv2.WINDOW_NORMAL)
    frame_time = 1.0 / fps
    t_start = time.perf_counter()
    paused = False

    print(f"  Playback: {n_pred} frames at {fps} fps. Press Q/ESC to quit, SPACE to pause.")

    for i in range(n_pred):
        src_idx = start_frame + int(i * step)
        if src_idx >= len(decoder):
            break

        frame = decoder[src_idx]  # [H, W, C] uint8 CUDA

        # VR crop
        if vr_mode:
            half_w = frame.shape[1] // 2
            if sbs_crop == "left":
                frame = frame[:, :half_w]
            else:
                frame = frame[:, half_w:]

        frame_np = frame.cpu().numpy()
        h, w = frame_np.shape[:2]
        scale = display_height / h
        disp_w = int(w * scale)
        frame_disp = cv2.resize(frame_np, (disp_w, display_height))
        frame_disp = cv2.cvtColor(frame_disp, cv2.COLOR_RGB2BGR)

        # Build prediction graph
        graph = np.zeros((graph_height, disp_w, 3), dtype=np.uint8)
        graph[:] = (30, 30, 30)

        # Show 10s window centered on current frame
        window = int(10 * fps)
        view_start = max(0, i - window // 2)
        view_end = min(n_pred, view_start + window)
        if view_end - view_start < window:
            view_start = max(0, view_end - window)

        view_preds = predictions[view_start:view_end]
        n_view = len(view_preds)

        # Draw grid lines
        for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gy = int((1 - g) * (graph_height - 20)) + 10
            cv2.line(graph, (0, gy), (disp_w, gy), (60, 60, 60), 1)

        # Draw prediction curve
        if n_view > 1:
            pts = []
            for j in range(n_view):
                x = int(j / (n_view - 1) * (disp_w - 1))
                y = int((1 - view_preds[j]) * (graph_height - 20)) + 10
                pts.append((x, y))
            for j in range(1, len(pts)):
                cv2.line(graph, pts[j - 1], pts[j], (0, 180, 255), 2)

        # Draw playback cursor
        if n_view > 1:
            cursor_x = int((i - view_start) / (n_view - 1) * (disp_w - 1))
        else:
            cursor_x = disp_w // 2
        cv2.line(graph, (cursor_x, 0), (cursor_x, graph_height), (0, 255, 0), 2)

        # Info text
        pos_val = predictions[i]
        info = f"pos={pos_val:.2f}  frame={i}/{n_pred}  time={i / fps:.1f}s"
        cv2.putText(graph, info, (10, graph_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        combined = np.vstack([frame_disp, graph])
        cv2.imshow("Prediction Playback", combined)

        # Real-time pacing
        if not paused:
            elapsed = time.perf_counter() - t_start
            target = (i + 1) * frame_time
            wait_ms = max(1, int((target - elapsed) * 1000))
        else:
            wait_ms = 50

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            if not paused:
                t_start = time.perf_counter() - i * frame_time

    cv2.destroyAllWindows()


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
    parser.add_argument("--playback", action="store_true",
                        help="Real-time video playback with prediction overlay")
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

    # Playback
    if args.playback and args.video is not None:
        playback_with_prediction(
            args.video, predictions, args.fps,
            vr_mode=args.vr_mode, sbs_crop=args.sbs_crop,
            start_time=args.start_time or 0.0,
        )

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
