"""Optical flow computation for motion features.

Computes dense optical flow and summarizes into compact feature vectors per frame.
Supports: Farneback (CPU, legacy), RAFT (GPU, recommended).
"""

import logging
from typing import Literal

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


# ── RAFT (GPU) ───────────────────────────────────────────────────────────────

_raft_model = None
_raft_device = None


def _load_raft(device: str = "cuda") -> torch.nn.Module:
    """Load RAFT-Small model from torchvision (cached)."""
    global _raft_model, _raft_device
    if _raft_model is not None and _raft_device == device:
        return _raft_model

    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights)
    model = model.to(device).eval()
    _raft_model = model
    _raft_device = device
    log.info("Loaded RAFT-Small on %s", device)
    return model


def compute_flow_raft(
    frames: np.ndarray,
    output_features: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Compute optical flow features using RAFT on GPU.

    Args:
        frames: [N, H, W, C] uint8 RGB array.
        output_features: Number of summary features per frame.
        device: CUDA device.

    Returns:
        np.ndarray of shape [N, output_features] float32.
    """
    n_frames = len(frames)
    if n_frames < 2:
        return np.zeros((n_frames, output_features), dtype=np.float32)

    model = _load_raft(device)
    features = np.zeros((n_frames, output_features), dtype=np.float32)

    # RAFT expects [B, 3, H, W] float32 tensors in [0, 1]
    # Process pairs of frames
    with torch.no_grad():
        for i in range(1, n_frames):
            img1 = torch.from_numpy(frames[i - 1]).permute(2, 0, 1).float().unsqueeze(0).to(device)  # [1, 3, H, W]
            img2 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(device)

            # RAFT returns list of flow predictions (iterative refinement), take last
            flow_predictions = model(img1, img2)
            flow = flow_predictions[-1].squeeze(0).cpu().numpy()  # [2, H, W]

            # Convert to [H, W, 2] for _summarize_flow
            flow_hwc = np.transpose(flow, (1, 2, 0))
            features[i] = _summarize_flow(flow_hwc, output_features)

    if n_frames > 1:
        features[0] = features[1]

    return features


def compute_flow_raft_dense_batched(
    frames: np.ndarray,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute dense optical flow using RAFT with batched frame pairs.

    Args:
        frames: [N, H, W, C] uint8 RGB array.
        device: CUDA device string.
        batch_size: Number of frame pairs to process simultaneously.

    Returns:
        np.ndarray of shape [N, 2, H, W] float32 dense flow fields.
    """
    n_frames = len(frames)
    h, w = frames.shape[1], frames.shape[2]
    if n_frames < 2:
        return np.zeros((n_frames, 2, h, w), dtype=np.float32)

    model = _load_raft(device)
    all_flows = np.zeros((n_frames, 2, h, w), dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(1, n_frames, batch_size):
            batch_end = min(batch_start + batch_size, n_frames)
            b = batch_end - batch_start

            imgs1 = np.stack([frames[i - 1] for i in range(batch_start, batch_end)])
            imgs2 = np.stack([frames[i] for i in range(batch_start, batch_end)])

            t1 = torch.from_numpy(imgs1).permute(0, 3, 1, 2).float().to(device)  # [B, 3, H, W]
            t2 = torch.from_numpy(imgs2).permute(0, 3, 1, 2).float().to(device)

            use_amp = (str(device).startswith("cuda") or device == "cuda")
            with torch.amp.autocast("cuda", enabled=use_amp):
                flow_preds = model(t1, t2)
            flows = flow_preds[-1].cpu().float().numpy()  # [B, 2, H, W]

            del t1, t2, flow_preds

            all_flows[batch_start:batch_end] = flows

    if n_frames > 1:
        all_flows[0] = all_flows[1]

    return all_flows


def normalize_flow_components(flows: np.ndarray) -> np.ndarray:
    """Normalize flow by spatial dimensions to make values resolution-independent.

    Args:
        flows: [N, 2, H, W] float32 dense flow.

    Returns:
        [N, 2, H, W] float32 with flow_x / W and flow_y / H.
    """
    h, w = flows.shape[2], flows.shape[3]
    out = flows.copy()
    out[:, 0] /= w   # flow_x normalized by width
    out[:, 1] /= h   # flow_y normalized by height
    return out


def downsample_dense_flow(flows: np.ndarray, out_size: int) -> np.ndarray:
    """Downsample dense flow fields to a fixed spatial resolution.

    Uses area interpolation (averaging) to reduce spatial dimensions.
    Does NOT rescale magnitudes — the values remain in the same displacement unit.

    Args:
        flows: [N, 2, H, W] float32 dense flow.
        out_size: Target spatial dimension (out_size x out_size).

    Returns:
        [N, 2, out_size, out_size] float32.
    """
    n, c, h, w = flows.shape
    if h == out_size and w == out_size:
        return flows
    result = np.zeros((n, c, out_size, out_size), dtype=np.float32)
    for i in range(n):
        for ch in range(c):
            result[i, ch] = cv2.resize(
                flows[i, ch], (out_size, out_size),
                interpolation=cv2.INTER_AREA,
            )
    return result


def summarize_dense_flow(flows: np.ndarray, output_features: int) -> np.ndarray:
    """Summarize dense flow fields into compact per-frame feature vectors.

    Args:
        flows: [N, 2, H, W] float32 dense flow.
        output_features: Number of summary features per frame.

    Returns:
        [N, output_features] float32.
    """
    n = flows.shape[0]
    features = np.zeros((n, output_features), dtype=np.float32)
    for i in range(n):
        flow_hwc = flows[i].transpose(1, 2, 0)  # [H, W, 2]
        features[i] = _summarize_flow(flow_hwc, output_features)
    return features


def compute_flow_raft_batched(
    frames: np.ndarray,
    output_features: int = 64,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute optical flow features using RAFT with batched frame pairs.

    Processes multiple frame pairs simultaneously on GPU for much higher
    throughput compared to the sequential ``compute_flow_raft``.

    Args:
        frames: [N, H, W, C] uint8 RGB array.
        output_features: Number of summary features per frame.
        device: CUDA device string.
        batch_size: Number of frame pairs to process simultaneously.

    Returns:
        np.ndarray of shape [N, output_features] float32.
    """
    dense = compute_flow_raft_dense_batched(frames, device, batch_size)
    return summarize_dense_flow(dense, output_features)


def compute_flow_farneback(
    frames: np.ndarray,
    output_features: int = 64,
) -> np.ndarray:
    """Compute optical flow features using Farneback method.

    Args:
        frames: [N, H, W, C] uint8 RGB array.
        output_features: Number of summary features per frame.

    Returns:
        np.ndarray of shape [N, output_features] float32.
    """
    n_frames = len(frames)
    if n_frames < 2:
        return np.zeros((n_frames, output_features), dtype=np.float32)

    features = np.zeros((n_frames, output_features), dtype=np.float32)

    # Convert to grayscale for flow computation
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    for i in range(1, n_frames):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i - 1], gray_frames[i],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        features[i] = _summarize_flow(flow, output_features)

    # First frame gets the same features as the second
    if n_frames > 1:
        features[0] = features[1]

    return features


def _summarize_flow(flow: np.ndarray, n_features: int) -> np.ndarray:
    """Summarize a dense flow field into a compact feature vector.

    Uses a spatial grid approach: divide the frame into a grid and compute
    mean flow magnitude and direction per cell.
    """
    h, w = flow.shape[:2]
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    # Compute magnitude and angle
    mag = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x)

    # Spatial grid: divide into cells
    # We want n_features values, use n_features//4 grid cells × 4 stats
    n_cells = max(n_features // 4, 4)
    grid_side = int(np.ceil(np.sqrt(n_cells)))
    cell_h = h // grid_side
    cell_w = w // grid_side

    features = []
    for row in range(grid_side):
        for col in range(grid_side):
            if len(features) >= n_cells:
                break
            r_start = row * cell_h
            r_end = min((row + 1) * cell_h, h)
            c_start = col * cell_w
            c_end = min((col + 1) * cell_w, w)

            cell_mag = mag[r_start:r_end, c_start:c_end]
            cell_fx = flow_x[r_start:r_end, c_start:c_end]
            cell_fy = flow_y[r_start:r_end, c_start:c_end]

            features.extend([
                np.mean(cell_mag),
                np.std(cell_mag),
                np.mean(cell_fx),
                np.mean(cell_fy),
            ])

    # Also add global statistics
    features.extend([
        np.mean(mag),
        np.std(mag),
        np.mean(flow_x),
        np.mean(flow_y),
        np.percentile(mag, 90),
        np.mean(angle),
    ])

    features = np.array(features, dtype=np.float32)

    # Pad or truncate to target size
    if len(features) >= n_features:
        return features[:n_features]
    else:
        padded = np.zeros(n_features, dtype=np.float32)
        padded[:len(features)] = features
        return padded


def compute_flow_for_video(
    frames_iterator,
    output_features: int = 64,
    batch_size: int = 120,
    method: Literal["farneback", "raft"] = "raft",
    device: str = "cuda",
    dense: bool = False,
    dense_size: int = 32,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute optical flow features for a full video via batched processing.

    Args:
        frames_iterator: Iterable yielding (frame_idx, frame_rgb_hwc) tuples.
        output_features: Feature vector dimension per frame.
        batch_size: Number of frames to process at once.
        method: "raft" (GPU, recommended) or "farneback" (CPU, legacy).
        device: CUDA device for RAFT.
        dense: If True and method is "raft", also return dense flow maps.
        dense_size: Spatial resolution for dense flow maps.

    Returns:
        If dense=False: np.ndarray of shape [N_frames, output_features].
        If dense=True: tuple of (summary [N, output_features], dense [N, 2, dense_size, dense_size]).
    """
    use_dense = dense and method == "raft"

    if not use_dense:
        compute_fn = (
            lambda frames: compute_flow_raft(frames, output_features, device)
            if method == "raft"
            else lambda frames: compute_flow_farneback(frames, output_features)
        )
    all_summary = []
    all_dense = [] if use_dense else None
    batch = []
    prev_last_frame = None
    total_processed = 0

    for _idx, frame in frames_iterator:
        batch.append(frame)

        if len(batch) >= batch_size:
            if prev_last_frame is not None:
                compute_batch = np.stack([prev_last_frame] + batch)
            else:
                compute_batch = np.stack(batch)

            if use_dense:
                raw_dense = compute_flow_raft_dense_batched(compute_batch, device, batch_size=64)
                norm_dense = normalize_flow_components(raw_dense)
                summary = summarize_dense_flow(raw_dense, output_features)
                ds_dense = downsample_dense_flow(norm_dense, dense_size)
                if prev_last_frame is not None:
                    all_summary.append(summary[1:])
                    all_dense.append(ds_dense[1:])
                else:
                    all_summary.append(summary)
                    all_dense.append(ds_dense)
            else:
                feats = compute_fn(compute_batch)
                if prev_last_frame is not None:
                    all_summary.append(feats[1:])
                else:
                    all_summary.append(feats)

            total_processed += len(batch)
            if total_processed % (batch_size * 10) == 0:
                log.info("Flow: processed %d frames", total_processed)

            prev_last_frame = batch[-1]
            batch = []

    # Process remaining
    if batch:
        if prev_last_frame is not None:
            compute_batch = np.stack([prev_last_frame] + batch)
        else:
            compute_batch = np.stack(batch)

        if use_dense:
            raw_dense = compute_flow_raft_dense_batched(compute_batch, device, batch_size=64)
            norm_dense = normalize_flow_components(raw_dense)
            summary = summarize_dense_flow(raw_dense, output_features)
            ds_dense = downsample_dense_flow(norm_dense, dense_size)
            if prev_last_frame is not None:
                all_summary.append(summary[1:])
                all_dense.append(ds_dense[1:])
            else:
                all_summary.append(summary)
                all_dense.append(ds_dense)
        else:
            feats = compute_fn(compute_batch) if not use_dense else None
            if prev_last_frame is not None:
                all_summary.append(feats[1:])
            else:
                all_summary.append(feats)

    if not all_summary:
        empty_summary = np.empty((0, output_features), dtype=np.float32)
        if use_dense:
            return empty_summary, np.empty((0, 2, dense_size, dense_size), dtype=np.float32)
        return empty_summary

    summary_out = np.concatenate(all_summary, axis=0)
    if use_dense:
        dense_out = np.concatenate(all_dense, axis=0)
        return summary_out, dense_out
    return summary_out
