"""Predict funscript positions using a trained DispositionTCN model.

GPU-optimized pipeline: torchcodec decode → YOLO on GPU tensors → roi_align on GPU
→ DispositionTCN on GPU.  Falls back to CPU paths when CUDA is unavailable.

Usage:
    # From a scene with pre-extracted spatial features:
    python scripts/predict_disposition.py --scene scene_00018_t01739_40s --plot

    # From a raw video (extracts spatial features on the fly):
    python scripts/predict_disposition.py --video path/to/video.mp4 --out out.funscript

    # Live playback with real-time prediction overlay:
    python scripts/predict_disposition.py --video path/to/video.mp4 --playback

    # Benchmark GPU vs baseline path:
    python scripts/predict_disposition.py --video path/to/video.mp4 --benchmark

    # Use a specific checkpoint:
    python scripts/predict_disposition.py --scene scene_00018_t01739_40s \
        --checkpoint data/models/checkpoints_disposition/best_disposition.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.dispositiontcn import DispositionTCN, extract_disposition_config
from src.data.pose import load_pose_model
from src.data.decode import stream_video_gpu, _has_torchcodec_cuda
from src.data.extraction import BACKBONE_STRIDE, _resolve_feature_layer, PARTNER_CLASS
from torchvision.ops import roi_align

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_MODEL_NAME = "vrlens-finetunes-multiclass-v2-yolo26m-pose"
DEFAULT_CHECKPOINT = Path("data/models/checkpoints_disposition/best_disposition.pt")


# ── GPU-native spatial extractor ──────────────────────────────────────────────

class GpuSpatialExtractor:
    """Extract spatial RoI feature grids with full GPU residency.

    Differences from scripts/extract_spatial.SpatialExtractor:
      - Accepts CUDA tensor input [N, H, W, C] uint8 directly from torchcodec.
      - Keeps roi_align outputs on GPU as float16 tensors.
      - Returns (spatial_gpu [N,1,C,R,R] float16 tensor, conf [N,1] float32 numpy).

    Passing CUDA tensors to Ultralytics predict() avoids the CPU round-trip.
    """

    def __init__(
        self,
        model,
        roi_output_size: int = 7,
        confidence_threshold: float = 0.02,
        device: str = "cuda",
    ):
        self.model = model
        self.roi_output_size = roi_output_size
        self.conf_threshold = confidence_threshold
        self.device = device
        self._features: torch.Tensor | None = None
        self.embed_dim: int | None = None

        self.layer_idx, self.layer_name = _resolve_feature_layer(model, None)
        self._hook = model.model.model[self.layer_idx].register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        self._features = output  # stays on GPU

    def _run_predict(self, frames):
        """Run YOLO predict; accepts CUDA tensor or list of numpy arrays."""
        self._features = None
        if isinstance(frames, torch.Tensor):
            # torchcodec yields [N, H, W, C] uint8 — YOLO LoadTensor needs [N, C, H, W] float32 in [0,1]
            frames = frames.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        results = self.model.predict(
            frames,
            verbose=False,
            save=False,
            conf=self.conf_threshold,
            iou=0.97,
        )
        return results, self._features

    def _extract_rois(self, n_frames, frame_h, frame_w, features, results):
        """Shared roi_align logic. Returns (spatial_gpu float16, conf_np float32)."""
        R = self.roi_output_size
        C = int(features.shape[1])
        self.embed_dim = C

        spatial_out = torch.zeros(
            (n_frames, 1, C, R, R), dtype=torch.float16, device=features.device
        )
        conf_out = np.zeros((n_frames, 1), dtype=np.float32)

        for i, result in enumerate(results):
            performer_box_xyxy = None
            performer_conf = 0.0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                det_conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                partner_idx = np.where(cls == PARTNER_CLASS)[0]
                if len(partner_idx) > 0:
                    best = partner_idx[int(np.argmax(det_conf[partner_idx]))]
                    performer_box_xyxy = boxes.xyxy[best]
                    performer_conf = float(det_conf[best])

            if performer_box_xyxy is not None:
                x1 = float(performer_box_xyxy[0])
                y1 = float(performer_box_xyxy[1])
                x2 = float(performer_box_xyxy[2])
                y2 = float(frame_h)
                conf_out[i, 0] = performer_conf
            else:
                x1 = 0.25 * frame_w
                y1 = 0.50 * frame_h
                x2 = 0.75 * frame_w
                y2 = float(frame_h)

            x1 = max(0.0, min(float(frame_w), x1))
            y1 = max(0.0, min(float(frame_h), y1))
            x2 = max(0.0, min(float(frame_w), x2))
            y2 = max(0.0, min(float(frame_h), y2))

            box_tensor = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32, device=features.device
            )
            roi_feat = roi_align(
                features[i:i + 1].float(),
                [box_tensor],
                output_size=R,
                spatial_scale=1.0 / BACKBONE_STRIDE,
                aligned=True,
            )  # [1, C, R, R] float32 on GPU
            spatial_out[i, 0] = roi_feat[0].to(torch.float16)

        return spatial_out, conf_out

    def extract_batch_gpu(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Extract RoI features from a CUDA uint8 tensor [N, H, W, C].

        Returns:
            spatial: [N, 1, C, R, R] float16 CUDA tensor
            conf:    [N, 1] float32 numpy array
        """
        n_frames = frames.shape[0]
        frame_h, frame_w = int(frames.shape[1]), int(frames.shape[2])
        results, features = self._run_predict(frames)
        if features is None:
            raise RuntimeError("No feature map captured")
        return self._extract_rois(n_frames, frame_h, frame_w, features, results)

    def extract_batch_numpy(
        self,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU-fallback: numpy frames [N,H,W,C] → numpy outputs."""
        n_frames = len(frames)
        frame_h, frame_w = frames.shape[1], frames.shape[2]
        results, features = self._run_predict(list(frames))
        if features is None:
            raise RuntimeError("No feature map captured")
        spatial_gpu, conf_np = self._extract_rois(
            n_frames, frame_h, frame_w, features, results
        )
        return spatial_gpu.cpu().numpy(), conf_np

    def close(self):
        if hasattr(self, "_hook"):
            self._hook.remove()
        self._features = None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: Path, device: torch.device,
) -> tuple[DispositionTCN, dict, dict]:
    """Load a DispositionTCN from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    data_cfg = ckpt.get("data_config", {})
    model = DispositionTCN(**extract_disposition_config(cfg))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    val_loss = ckpt.get("val_loss")
    print(
        f"Loaded DispositionTCN: epoch {ckpt.get('epoch', '?')}, "
        + (f"val_loss={val_loss:.6f}" if isinstance(val_loss, (int, float)) else f"val_loss={val_loss}")
    )
    return model, cfg, data_cfg


# ── GPU-accelerated feature extraction ───────────────────────────────────────

def extract_spatial_gpu(
    video_path: Path,
    model_name: str,
    device: torch.device,
    roi_size: int = 7,
    batch_size: int = 32,
    vr_mode: bool = False,
    sbs_crop: str = "left",
    target_fps: float = 30.0,
    start_time: float | None = None,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated spatial feature extraction.

    Pipeline: torchcodec (CUDA) → YOLO on GPU tensor → roi_align on GPU.
    Falls back to CPU/numpy for each step if CUDA is unavailable.

    Returns:
        spatial: [T, 1, C, roi_size, roi_size] float16 numpy
        conf:    [T, 1] float32 numpy
    """
    model_path = Path("data/models/pose") / f"{model_name}.pt"
    pose_model = load_pose_model(
        model_name=model_name,
        model_path=str(model_path),
        device=str(device),
    )
    extractor = GpuSpatialExtractor(
        pose_model,
        roi_output_size=roi_size,
        device=str(device),
    )

    crop_left_half = vr_mode and sbs_crop == "left"
    timings = {"video_decode": 0.0, "yolo_roi": 0.0}

    all_spatial: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    n_frames = 0

    # Estimate total frames for progress bar
    total_est = None
    try:
        from torchcodec.decoders import VideoDecoder as _VD
        _tmp = _VD(str(video_path))
        src_fps = _tmp.metadata.average_fps or 30.0
        src_n = _tmp.metadata.num_frames or 0
        _tmp = None
        eff = target_fps or src_fps
        if duration:
            total_est = int(duration * eff)
        elif src_fps > 0:
            total_est = int(src_n * eff / src_fps)
    except Exception:
        pass

    chunk_size = batch_size
    tp = tqdm(
        total=max(1, total_est // chunk_size) if total_est else None,
        unit=f"×{chunk_size}f", desc="GPU extract", dynamic_ncols=True,
    )

    use_gpu_path = device.type == "cuda" and _has_torchcodec_cuda()
    t_last = time.perf_counter()

    for chunk_data in stream_video_gpu(
        video_path,
        device=str(device),
        crop_left_half=crop_left_half,
        target_size=640,
        target_fps=target_fps,
        start_time=start_time,
        duration=duration,
        chunk_size=chunk_size,
        as_numpy=not use_gpu_path,
    ):
        t_decode_end = time.perf_counter()
        timings["video_decode"] += t_decode_end - t_last

        t0 = time.perf_counter()
        if use_gpu_path and isinstance(chunk_data, torch.Tensor):
            # GPU path: CUDA tensor → YOLO → roi_align on GPU
            spatial_gpu, conf_np = extractor.extract_batch_gpu(chunk_data)
            spatial_np = spatial_gpu.cpu().numpy()
            del spatial_gpu
        else:
            # CPU fallback
            if isinstance(chunk_data, torch.Tensor):
                chunk_np = chunk_data.cpu().numpy()
            elif isinstance(chunk_data, list):
                chunk_np = np.stack(chunk_data)
            else:
                chunk_np = chunk_data
            spatial_np, conf_np = extractor.extract_batch_numpy(chunk_np)

        timings["yolo_roi"] += time.perf_counter() - t0
        all_spatial.append(spatial_np)
        all_conf.append(conf_np)
        n_frames += len(spatial_np)
        t_last = time.perf_counter()
        tp.update(1)

    tp.close()
    extractor.close()

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  Processed {n_frames} frames total")
    total = sum(timings.values())
    print(f"\n  Timing breakdown:")
    for k, v in timings.items():
        print(f"    {k:20s}: {v:6.2f}s ({v / total * 100:4.1f}%)")
    print(f"    {'TOTAL':20s}: {total:6.2f}s")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2
        print(f"  Peak VRAM allocated: {peak_mb:.0f} MB  (reserved: {reserved_mb:.0f} MB)")

    return np.concatenate(all_spatial, axis=0), np.concatenate(all_conf, axis=0)


# ── Sliding-window prediction ─────────────────────────────────────────────────


def sliding_window_predict(model: DispositionTCN,
    spatial: np.ndarray,
    conf: np.ndarray,
    device: torch.device,
    seq_len: int = 120,
    stride: int = 60,
) -> np.ndarray:
    """Slide a window over the sequence and average overlapping predictions."""
    n_frames = len(spatial)
    pred_sum = np.zeros(n_frames, dtype=np.float32)
    pred_count = np.zeros(n_frames, dtype=np.float32)

    starts = list(range(0, n_frames - seq_len + 1, stride))
    if n_frames >= seq_len and starts and starts[-1] + seq_len < n_frames:
        starts.append(n_frames - seq_len)

    with torch.no_grad():
        for start in starts:
            end = start + seq_len
            sp = torch.from_numpy(spatial[start:end]).float().unsqueeze(0).to(device)
            co = torch.from_numpy(conf[start:end]).float().unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(sp, co)

            p = out[0].float().cpu().numpy()
            weight = np.bartlett(seq_len).astype(np.float32) + 0.01
            pred_sum[start:end] += p * weight
            pred_count[start:end] += weight

    if n_frames < seq_len:
        sp = torch.from_numpy(spatial).float().unsqueeze(0).to(device)
        co = torch.from_numpy(conf).float().unsqueeze(0).to(device)
        pad = seq_len - n_frames
        sp = F.pad(sp, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad))
        co = F.pad(co, (0, 0, 0, pad))
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(sp, co)
        return out[0, :n_frames].float().cpu().numpy()

    mask = pred_count > 0
    pred_sum[mask] /= pred_count[mask]
    return pred_sum


def predictions_to_funscript(
    positions: np.ndarray, fps: float = 30.0, start_time: float = 0.0,
) -> dict:
    """Convert per-frame positions [0,1] to funscript JSON."""
    actions = []
    for i, pos in enumerate(positions):
        at_ms = int(round((i / fps + start_time) * 1000.0))
        pos_int = max(0, min(100, int(round(float(pos) * 100.0))))
        actions.append({"at": at_ms, "pos": pos_int})
    return {"version": "1.0", "inverted": False, "range": 100, "actions": actions}


# ── Live playback with real-time prediction ───────────────────────────────────

def live_playback_with_prediction(
    video_path: Path,
    model_name: str,
    model: DispositionTCN,
    device: torch.device,
    roi_size: int = 7,
    vr_mode: bool = False,
    sbs_crop: str = "left",
    start_time: float | None = None,
    duration: float | None = None,
    target_fps: float = 30.0,
    seq_len: int = 120,
    stride: int = 60,
) -> np.ndarray:
    """Live video playback with simultaneous DispositionTCN prediction.

    Runs decode → YOLO → roi_align → DispositionTCN on a background thread
    while displaying the video in real time via a tkinter GUI.

    Returns the final predictions array.
    """
    import os
    import signal
    import threading
    import tkinter as tk
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from PIL import Image, ImageTk

    MAX_FRAME_BUF = 60000
    STATS_EVERY_N = 8
    GRAPH_EVERY_N = 4

    _lock = threading.Lock()
    _state: dict = {
        "frames":           {},
        "predictions":      None,
        "decode_fps":       0.0,
        "yolo_fps":         0.0,
        "frames_decoded":   0,
        "frames_predicted": 0,
        "total_est":        0,
        "status":           "Initializing…",
        "done":             False,
        "error":            None,
    }

    _running = [True]

    def _worker() -> None:
        try:
            with _lock:
                _state["status"] = "Loading pose model…"

            model_path = Path("data/models/pose") / f"{model_name}.pt"
            pose_model = load_pose_model(
                model_name=model_name,
                model_path=str(model_path),
                device=str(device),
            )
            extractor = GpuSpatialExtractor(
                pose_model,
                roi_output_size=roi_size,
                device=str(device),
            )

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
            use_gpu_path = device.type == "cuda" and _has_torchcodec_cuda()

            all_spatial: list[np.ndarray] = []
            all_conf:    list[np.ndarray] = []
            frame_count = 0
            t0_total = time.perf_counter()
            t_yolo   = 0.0
            n_yolo   = 0

            for chunk_data in stream_video_gpu(
                video_path,
                device=str(device),
                crop_left_half=crop_left,
                target_size=640,
                target_fps=target_fps,
                start_time=start_time,
                duration=duration,
                chunk_size=batch_size,
                as_numpy=not use_gpu_path,
            ):
                if not _running[0]:
                    break

                # Always get numpy for display
                if isinstance(chunk_data, torch.Tensor):
                    chunk_np = chunk_data.cpu().numpy()
                elif isinstance(chunk_data, list):
                    chunk_np = np.stack(chunk_data)
                else:
                    chunk_np = chunk_data
                chunk_len = len(chunk_np)

                # Suspend strategy: wait for buffer room
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

                # Feature extraction
                t1 = time.perf_counter()
                if use_gpu_path and isinstance(chunk_data, torch.Tensor):
                    spatial_gpu, conf_np = extractor.extract_batch_gpu(chunk_data)
                    spatial_np = spatial_gpu.cpu().numpy()
                    del spatial_gpu
                else:
                    spatial_np, conf_np = extractor.extract_batch_numpy(chunk_np)

                t_yolo += time.perf_counter() - t1
                n_yolo += chunk_len
                all_spatial.append(spatial_np)
                all_conf.append(conf_np)

                frame_count += chunk_len
                elapsed = time.perf_counter() - t0_total
                decode_fps = frame_count / elapsed if elapsed > 0 else 0.0
                yolo_fps   = n_yolo / t_yolo if t_yolo > 0 else 0.0

                preds_new = None
                if frame_count >= seq_len:
                    sp_a = np.concatenate(all_spatial)
                    co_a = np.concatenate(all_conf)
                    n = min(len(sp_a), len(co_a))
                    preds_new = sliding_window_predict(
                        model, sp_a[:n], co_a[:n], device, seq_len, stride,
                    )

                with _lock:
                    _state["decode_fps"]     = decode_fps
                    _state["yolo_fps"]       = yolo_fps
                    _state["frames_decoded"] = frame_count
                    if preds_new is not None:
                        _state["predictions"]      = preds_new
                        _state["frames_predicted"] = len(preds_new)

            # Final prediction pass
            sp_a = np.concatenate(all_spatial)
            co_a = np.concatenate(all_conf)
            n = min(len(sp_a), len(co_a))
            preds_final = sliding_window_predict(
                model, sp_a[:n], co_a[:n], device, seq_len, stride,
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

    # ── Position meter overlay ─────────────────────────────────────────────
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

    # ── tkinter GUI ────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("DispositionTCN — Live Prediction")
    root.configure(bg="#1e1e1e")
    root.resizable(True, True)

    _tk_photo: list = [None]
    _pos      = [0]
    _playing  = [False]
    _speed    = [1.0]
    _t_start  = [time.perf_counter()]
    _f_start  = [0]
    _buf_wait = [True]
    _tick     = [0]

    BUF_START_THRESHOLD  = 360
    BUF_RESUME_THRESHOLD = 360
    BUF_LOW_THRESHOLD    = 30
    frame_delay_ms = max(16, int(round(1000.0 / target_fps)))

    top_row = tk.Frame(root, bg="#1e1e1e")
    top_row.pack(fill=tk.BOTH, expand=False, padx=4, pady=(4, 0))

    video_canvas = tk.Canvas(top_row, bg="#000000", width=640, height=640, highlightthickness=0)
    video_canvas.pack(side=tk.LEFT, padx=(0, 4))

    stats_outer = tk.Frame(top_row, bg="#252526", width=230)
    stats_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
    stats_outer.pack_propagate(False)
    tk.Label(stats_outer, text="Live Stats", bg="#252526", fg="#cccccc",
             font=("Segoe UI", 10, "bold"), anchor="w").pack(fill=tk.X, padx=8, pady=(8, 4))

    stats_text = tk.Text(stats_outer, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9),
                         borderwidth=0, highlightthickness=0, state=tk.DISABLED,
                         wrap=tk.WORD, height=22, width=26)
    stats_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    stats_text.tag_configure("head", foreground="#4ec9b0", font=("Consolas", 9, "bold"))
    stats_text.tag_configure("val",  foreground="#ce9178")
    stats_text.tag_configure("ok",   foreground="#6a9955")
    stats_text.tag_configure("warn", foreground="#d79921")
    stats_text.tag_configure("err",  foreground="#f44747")
    stats_text.tag_configure("dim",  foreground="#555555")

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
    [pred_line]   = ax.plot([], [], lw=1.5, color="#e37933", label="Prediction", animated=True)
    [cursor_line] = ax.plot([], [], lw=1.5, color="#4ee344", linestyle="--", label="Now", animated=True)
    ax.legend(fontsize=7, labelcolor="#888888", facecolor="#252526", edgecolor="#444444", loc="upper left")
    fig.tight_layout(pad=0.6)

    timeline_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    timeline_canvas.get_tk_widget().pack(fill=tk.X)
    timeline_canvas.draw()
    _graph_bg = [timeline_canvas.copy_from_bbox(ax.bbox)]

    def _on_timeline_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        target_frame = max(0, int(event.xdata * target_fps))
        _pos[0] = target_frame
        _t_start[0] = time.perf_counter()
        _f_start[0] = target_frame
        _buf_wait[0] = False

    timeline_canvas.mpl_connect("button_press_event", _on_timeline_click)

    _seek_var = tk.IntVar(value=0)
    _seek_updating = [False]

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
    seek_slider = tk.Scale(seek_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                           variable=_seek_var, command=_on_seek_slider,
                           bg="#1e1e1e", fg="#888888", troughcolor="#3c3c3c",
                           highlightthickness=0, showvalue=False, length=600,
                           sliderrelief=tk.FLAT, sliderlength=12)
    seek_slider.pack(fill=tk.X, expand=True)
    seek_time_label = tk.Label(seek_frame, text="0:00 / 0:00",
                               bg="#1e1e1e", fg="#888888", font=("Consolas", 8))
    seek_time_label.pack()

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
        _buf_wait[0] = False

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
        tk.Radiobutton(ctrl_frame, text=f"{spd}×", variable=speed_var, value=spd,
                       bg="#1e1e1e", fg="#cccccc", selectcolor="#1e1e1e",
                       activebackground="#1e1e1e", font=("Segoe UI", 9),
                       command=_set_spd).pack(side=tk.LEFT, padx=2)

    worker_status_label = tk.Label(ctrl_frame, text="● Starting…",
                                   bg="#1e1e1e", fg="#d79921", font=("Segoe UI", 9))
    worker_status_label.pack(side=tk.RIGHT, padx=8)

    _after_id = [None]

    def _on_close() -> None:
        _running[0] = False
        if _after_id[0] is not None:
            try:
                root.after_cancel(_after_id[0])
            except Exception:
                pass
            _after_id[0] = None
        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.bind("<space>",   lambda e: _toggle_play())
    root.bind("<Left>",    lambda e: _seek(-int(target_fps)))
    root.bind("<Right>",   lambda e: _seek(int(target_fps)))
    root.bind("<q>",       lambda e: _on_close())
    root.bind("<Escape>",  lambda e: _on_close())

    def _sigint_handler(sig, frame) -> None:
        _running[0] = False
        try:
            root.after(0, _on_close)
        except Exception:
            os._exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    _prev_xlim = [0.0, 30.0]

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
        row("Playing:", "Yes" if _playing[0] else "No",
            "ok" if _playing[0] else "warn")
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

    def _update_graph(preds: np.ndarray, cur_frame: int) -> None:
        if preds is None or len(preds) == 0:
            return
        t_all = np.arange(len(preds)) / target_fps
        t_cur = cur_frame / target_fps
        half  = 15.0
        t0w   = max(0.0, t_cur - half)
        t1w   = t0w + half * 2
        if t1w > t_all[-1]:
            t1w = t_all[-1]
            t0w = max(0.0, t1w - half * 2)
        mask = (t_all >= t0w) & (t_all <= t1w)
        pred_line.set_data(t_all[mask], preds[mask])
        cursor_line.set_data([t_cur, t_cur], [-0.05, 1.05])

        new_xlim = [t0w, t1w]
        if abs(new_xlim[0] - _prev_xlim[0]) > 0.5 or abs(new_xlim[1] - _prev_xlim[1]) > 0.5:
            ax.set_xlim(t0w, t1w)
            _prev_xlim[:] = new_xlim
            timeline_canvas.draw()
            _graph_bg[0] = timeline_canvas.copy_from_bbox(ax.bbox)

        timeline_canvas.restore_region(_graph_bg[0])
        ax.draw_artist(pred_line)
        ax.draw_artist(cursor_line)
        timeline_canvas.blit(ax.bbox)

    def _tick_display() -> None:
        if not _running[0]:
            return
        try:
            _tick_display_inner()
        except Exception:
            pass

    def _tick_display_inner() -> None:
        with _lock:
            st     = dict(_state)
            frames = dict(_state["frames"])
            preds  = _state["predictions"]

        n_decoded = st["frames_decoded"]
        n_pred    = len(preds) if preds is not None else 0

        if _playing[0]:
            elapsed_s  = (time.perf_counter() - _t_start[0]) * _speed[0]
            target_idx = int(_f_start[0] + elapsed_s * target_fps)
        else:
            target_idx = _pos[0]

        buf_sz      = len(frames)
        worker_done = st["done"]

        if _buf_wait[0]:
            needed = BUF_START_THRESHOLD if _pos[0] == 0 else BUF_RESUME_THRESHOLD
            if buf_sz >= needed or worker_done:
                _buf_wait[0] = False
                if not _playing[0]:
                    _playing[0] = True
                    _t_start[0] = time.perf_counter()
                    _f_start[0] = _pos[0]
                    btn_play.config(text="⏸ Pause")
                    elapsed_s  = (time.perf_counter() - _t_start[0]) * _speed[0]
                    target_idx = int(_f_start[0] + elapsed_s * target_fps)
        elif not worker_done:
            if buf_sz < BUF_LOW_THRESHOLD or (frames and target_idx > max(frames.keys())):
                _buf_wait[0] = True
                _playing[0] = False
                btn_play.config(text="▶ Play")
                _t_start[0] = time.perf_counter()
                _f_start[0] = _pos[0]
                target_idx = _pos[0]

        _pos[0] = target_idx

        if st["done"] and n_pred > 0 and _pos[0] >= n_pred - 1:
            _playing[0] = False
            btn_play.config(text="▶ Play")

        frame_rgb = frames.get(_pos[0])
        if frame_rgb is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            pred_val: float | None = None
            if preds is not None and len(preds) > 0:
                idx = min(_pos[0], len(preds) - 1)
                pred_val = float(preds[idx])
            if pred_val is not None:
                frame_bgr = _draw_position_meter(frame_bgr, pred_val)
                pos_label.config(text=f"pos={pred_val:.3f}")
            else:
                pos_label.config(text="pos=…")

            if _buf_wait[0]:
                cv2.putText(frame_bgr, "BUFFERING…", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

            img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=img)
            _tk_photo[0] = photo
            h_f, w_f = frame_bgr.shape[:2]
            video_canvas.config(width=w_f, height=h_f)
            video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        if _playing[0] and _pos[0] > 30:
            evict_below = _pos[0] - 30
            with _lock:
                to_del = [k for k in _state["frames"] if k < evict_below]
                for k in to_del:
                    del _state["frames"][k]

        t_sec = _pos[0] / max(target_fps, 1.0)
        frame_label.config(text=f"Frame: {_pos[0]} / {n_decoded}  |  {t_sec:.1f}s")

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

        _tick[0] += 1
        if _tick[0] % STATS_EVERY_N == 0:
            _update_stats(st, _pos[0])
        if _tick[0] % GRAPH_EVERY_N == 0:
            _update_graph(preds, _pos[0])

        try:
            _after_id[0] = root.after(frame_delay_ms, _tick_display)
        except Exception:
            pass

    _after_id[0] = root.after(100, _tick_display)
    root.mainloop()

    _running[0] = False
    worker_thread.join(timeout=5.0)

    with _lock:
        final_preds = _state["predictions"]
        err         = _state["error"]

    if err is not None:
        print(f"Warning: worker thread encountered an error: {err}")

    result = final_preds if final_preds is not None else np.array([], dtype=np.float32)

    if worker_thread.is_alive():
        print("Worker thread still running — forcing exit.")
        os._exit(0)

    return result


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(
    video_path: Path,
    model: DispositionTCN,
    model_cfg: dict,
    data_cfg: dict,
    device: torch.device,
    seq_len: int,
    stride: int,
) -> None:
    """Benchmark GPU path vs CPU/baseline path and compare prediction accuracy."""
    from scripts.extract_spatial import SpatialExtractor, load_video_frames

    model_name = data_cfg.get("model_name", DEFAULT_MODEL_NAME)
    roi_size   = model_cfg.get("roi_size", 7)

    print("\n" + "=" * 60)
    print("BENCHMARK: GPU path vs baseline path")
    print("=" * 60)

    # ── GPU path ──────────────────────────────────────────────────────────
    print("\n[1/2] GPU path (torchcodec + YOLO on CUDA tensor + GPU roi_align)…")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_gpu_start = time.perf_counter()
    spatial_gpu, conf_gpu = extract_spatial_gpu(
        video_path, model_name, device, roi_size=roi_size,
    )
    t_gpu_extract = time.perf_counter() - t_gpu_start
    print(f"  GPU extract time : {t_gpu_extract:.2f}s")

    t0 = time.perf_counter()
    preds_gpu = sliding_window_predict(model, spatial_gpu, conf_gpu, device, seq_len, stride)
    t_gpu_pred = time.perf_counter() - t0
    t_gpu_total = t_gpu_extract + t_gpu_pred
    print(f"  GPU predict time : {t_gpu_pred:.2f}s")
    print(f"  GPU TOTAL        : {t_gpu_total:.2f}s  ({len(preds_gpu)} frames)")

    # ── Baseline path ─────────────────────────────────────────────────────
    print("\n[2/2] Baseline path (load_video_frames → numpy → YOLO)…")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_base_start = time.perf_counter()
    model_path = Path("data/models/pose") / f"{model_name}.pt"
    pose_model_b = load_pose_model(
        model_name=model_name,
        model_path=str(model_path),
        device=str(device),
    )
    extractor_b = SpatialExtractor(pose_model_b, roi_output_size=roi_size, device=str(device))
    frames = load_video_frames(video_path)
    all_sp, all_co = [], []
    for start in range(0, len(frames), 32):
        batch = frames[start:start + 32]
        sp, co = extractor_b.extract_batch(batch)
        all_sp.append(sp)
        all_co.append(co)
    extractor_b.close()
    spatial_base = np.concatenate(all_sp)
    conf_base    = np.concatenate(all_co)
    t_base_extract = time.perf_counter() - t_base_start
    print(f"  Baseline extract : {t_base_extract:.2f}s")

    t0 = time.perf_counter()
    preds_base = sliding_window_predict(model, spatial_base, conf_base, device, seq_len, stride)
    t_base_pred = time.perf_counter() - t0
    t_base_total = t_base_extract + t_base_pred
    print(f"  Baseline predict : {t_base_pred:.2f}s")
    print(f"  Baseline TOTAL   : {t_base_total:.2f}s  ({len(preds_base)} frames)")

    # ── Summary ───────────────────────────────────────────────────────────
    speedup = t_base_total / max(t_gpu_total, 1e-6)
    n = min(len(preds_gpu), len(preds_base))
    mse = float(np.mean((preds_gpu[:n] - preds_base[:n]) ** 2))
    max_diff = float(np.max(np.abs(preds_gpu[:n] - preds_base[:n])))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  GPU path total   : {t_gpu_total:7.2f}s")
    print(f"  Baseline total   : {t_base_total:7.2f}s")
    print(f"  Speedup          : {speedup:.2f}×")
    print(f"  Prediction MSE   : {mse:.8f}  (GPU vs baseline)")
    print(f"  Max abs diff     : {max_diff:.6f}")
    if mse < 1e-6:
        print("  ✓ Predictions are numerically identical")
    else:
        print("  ⚠  Small numerical difference (expected from float16 quantisation in roi)")
    print("=" * 60 + "\n")


# ── Plot helper ───────────────────────────────────────────────────────────────

def _plot_predictions(
    predictions: np.ndarray,
    labels: np.ndarray | None,
    fps: float,
    source_name: str,
    args,
) -> None:
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

    axes[0].plot(time_axis, predictions, lw=1.5, color="darkorange", label="Disposition prediction")
    if labels is not None:
        axes[0].plot(time_axis[:len(labels)], labels, lw=1.2, alpha=0.7,
                     color="steelblue", label="Ground truth")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Position")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"DispositionTCN Prediction — {source_name}")
    axes[0].grid(alpha=0.3)

    if labels is not None:
        n = min(len(predictions), len(labels))
        error = predictions[:n] - labels[:n]
        axes[1].fill_between(time_axis[:n], error, color="red", alpha=0.4,
                             label="Error (pred - target)")
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disposition with DispositionTCN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--scene", type=str, default=None,
                     help="Scene name (loads pre-extracted spatial features)")
    src.add_argument("--video", type=Path, default=None,
                     help="Raw video path (extracts features on the fly)")

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--out",        type=Path, default=None,
                        help="Output funscript path (auto-derived if omitted)")
    parser.add_argument("--device",     type=str,  default="auto")
    parser.add_argument("--seq-len",    type=int,  default=120)
    parser.add_argument("--stride",     type=int,  default=60)
    parser.add_argument("--model-name", type=str,  default=DEFAULT_MODEL_NAME)
    parser.add_argument("--roi-size",   type=int,  default=None,
                        help="Override roi_size from checkpoint")
    parser.add_argument("--fps",        type=float, default=30.0,
                        help="Target decoding FPS")

    parser.add_argument("--vr",         dest="vr",  action="store_true",  default=False,
                        help="VR / SBS video — crop a single eye before decode")
    parser.add_argument("--no-vr",      dest="vr",  action="store_false")
    parser.add_argument("--sbs-crop",   type=str,   default="left",
                        choices=["left", "right"],
                        help="Which SBS half to use when --vr is set")

    parser.add_argument("--playback",   action="store_true",
                        help="Open live tkinter playback window")
    parser.add_argument("--benchmark",  action="store_true",
                        help="Benchmark GPU path vs baseline")

    parser.add_argument("--plot",       action="store_true",
                        help="Show matplotlib prediction plot after inference")
    parser.add_argument("--save-plot",  type=Path, default=None,
                        help="Save prediction plot to this file")

    args = parser.parse_args()

    if args.scene is None and args.video is None:
        parser.error("Provide --scene or --video")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model, model_cfg, data_cfg = load_model(args.checkpoint, device)
    roi_size   = args.roi_size or model_cfg.get("roi_size", 7)
    model_name = data_cfg.get("model_name", args.model_name)

    # Determine video path
    if args.video is not None:
        video_path = args.video
    else:
        video_path = args.data_dir / "preprocessed" / f"{args.scene}.mp4"

    # ── Benchmark mode ────────────────────────────────────────────────────
    if args.benchmark:
        if args.video is None and args.scene is None:
            parser.error("--benchmark requires --video or --scene")
        run_benchmark(
            video_path=video_path,
            model=model,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            device=device,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        return

    # ── Live playback mode ────────────────────────────────────────────────
    if args.playback:
        predictions = live_playback_with_prediction(
            video_path=video_path,
            model_name=model_name,
            model=model,
            device=device,
            roi_size=roi_size,
            vr_mode=args.vr,
            sbs_crop=args.sbs_crop,
            target_fps=args.fps,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        if len(predictions) == 0:
            print("No predictions generated during playback.")
            return
    else:
        # ── Batch inference mode ──────────────────────────────────────────
        labels = None

        if args.scene and (args.data_dir / "processed" / args.scene / "spatial" / f"{model_name}.h5").exists():
            scene_dir    = args.data_dir / "processed" / args.scene
            spatial_path = scene_dir / "spatial" / f"{model_name}.h5"
            conf_path    = scene_dir / "spatial" / f"{model_name}_conf.h5"
            print(f"Loading spatial features from {spatial_path}")
            import h5py
            with h5py.File(str(spatial_path), "r") as hf:
                spatial = hf["spatial"][:]
            with h5py.File(str(conf_path), "r") as hf:
                conf = hf["conf"][:]
            label_path = scene_dir / "labels.npy"
            if label_path.exists():
                labels = np.load(str(label_path))
        else:
            print(f"Extracting features from video: {video_path}")
            spatial, conf = extract_spatial_gpu(
                video_path, model_name, device,
                roi_size=roi_size,
                vr_mode=args.vr,
                sbs_crop=args.sbs_crop,
                target_fps=args.fps,
            )

        print(f"Spatial features: {spatial.shape}, Confidence: {conf.shape}")

        t0 = time.perf_counter()
        predictions = sliding_window_predict(
            model, spatial, conf, device,
            seq_len=args.seq_len, stride=args.stride,
        )
        t_pred = time.perf_counter() - t0
        print(f"Prediction: {len(predictions)} frames in {t_pred:.2f}s")
        print(f"  mean={predictions.mean():.4f}  std={predictions.std():.4f}")

        if labels is not None:
            n = min(len(predictions), len(labels))
            mse = float(np.mean((predictions[:n] - labels[:n]) ** 2))
            print(f"  MSE vs ground truth: {mse:.6f}")

    # ── Save funscript ────────────────────────────────────────────────────
    out_path = args.out
    if out_path is None:
        # Auto-derive: <video_stem>.funscript alongside the video
        out_path = video_path.with_suffix(".funscript")

    funscript = predictions_to_funscript(predictions, fps=args.fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(funscript, fh)
    print(f"Saved funscript → {out_path}")

    # ── Optional plot ─────────────────────────────────────────────────────
    if args.plot or args.save_plot:
        _plot_predictions(
            predictions=predictions,
            labels=locals().get("labels"),
            fps=args.fps,
            source_name=str(args.scene or args.video),
            args=args,
        )


if __name__ == "__main__":
    main()

