"""Funscript file parsing and creation utilities.

Funscript format: JSON with an "actions" array of {at: ms, pos: 0-100} objects.
Internally we normalize positions to [0, 1] and work with frame indices.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline, interp1d

log = logging.getLogger(__name__)


def load_funscript(path: str | Path) -> dict:
    """Load a funscript file and return the raw JSON dict."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_actions(data: dict) -> list[tuple[int, int]]:
    """Extract (timestamp_ms, position_0_100) pairs from funscript data, sorted by time."""
    actions = data.get("actions", [])
    pairs = [(a["at"], a["pos"]) for a in actions]
    pairs.sort(key=lambda x: x[0])
    # remove first and last 10%
    n = len(pairs)
    if n > 2:
        start_idx = n // 10
        end_idx = n - start_idx
        pairs = pairs[start_idx:end_idx]

    return pairs


def actions_to_frame_labels(
    actions: list[tuple[int, int]],
    fps: float,
    total_frames: int,
    interpolation: str = "cubic",
) -> np.ndarray:
    """Convert funscript actions to per-frame position labels in [0, 1].

    Args:
        actions: List of (timestamp_ms, position_0_100) pairs.
        fps: Video frame rate.
        total_frames: Total number of frames in the video.
        interpolation: "linear", "cubic", or "nearest".

    Returns:
        np.ndarray of shape [total_frames] with values in [0, 1].
    """
    if not actions:
        return np.zeros(total_frames, dtype=np.float32)

    timestamps_ms = np.array([a[0] for a in actions], dtype=np.float64)
    positions = np.array([a[1] for a in actions], dtype=np.float64) / 100.0

    # Convert timestamps to frame indices
    frame_indices = timestamps_ms * fps / 1000.0

    # Create frame-level labels
    all_frames = np.arange(total_frames, dtype=np.float64)

    if interpolation == "cubic" and len(actions) >= 4:
        try:
            spline = make_interp_spline(frame_indices, positions, k=3)
            labels = spline(all_frames)
        except Exception:
            log.warning("Cubic spline failed, falling back to linear interpolation")
            interp_fn = interp1d(frame_indices, positions, kind="linear",
                                 fill_value="extrapolate", bounds_error=False)
            labels = interp_fn(all_frames)
    elif interpolation == "nearest":
        interp_fn = interp1d(frame_indices, positions, kind="nearest",
                             fill_value="extrapolate", bounds_error=False)
        labels = interp_fn(all_frames)
    else:
        interp_fn = interp1d(frame_indices, positions, kind="linear",
                             fill_value="extrapolate", bounds_error=False)
        labels = interp_fn(all_frames)

    labels = np.clip(labels, 0.0, 1.0).astype(np.float32)
    return labels


def frame_labels_to_funscript(
    labels: np.ndarray,
    fps: float,
    simplify: bool = True,
    tolerance: float = 2.0,
) -> dict:
    """Convert per-frame position labels back to funscript format.

    Args:
        labels: Array of shape [N] with values in [0, 1].
        fps: Video frame rate.
        simplify: If True, reduce points using RDP-like simplification (keep peaks/troughs).
        tolerance: Position tolerance for simplification (in 0-100 scale).

    Returns:
        Funscript JSON dict.
    """
    positions_100 = (labels * 100).astype(np.float64)
    n_frames = len(labels)

    if simplify:
        indices = _simplify_points(positions_100, tolerance)
    else:
        indices = list(range(n_frames))

    actions = []
    for idx in indices:
        timestamp_ms = int(round(idx * 1000.0 / fps))
        pos = int(round(np.clip(positions_100[idx], 0, 100)))
        actions.append({"at": timestamp_ms, "pos": pos})

    return {
        "version": "1.0",
        "inverted": False,
        "range": 100,
        "actions": actions,
        "metadata": {
            "creator": "VideoToMotion",
            "type": "basic",
        },
    }


def save_funscript(data: dict, path: str | Path) -> None:
    """Save a funscript dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Saved funscript to %s (%d actions)", path, len(data.get("actions", [])))


def _simplify_points(values: np.ndarray, tolerance: float) -> list[int]:
    """Keep peaks, troughs, and points where the curve changes significantly."""
    if len(values) <= 2:
        return list(range(len(values)))

    indices = [0]

    for i in range(1, len(values) - 1):
        prev_val = values[indices[-1]]
        curr_val = values[i]
        next_val = values[i + 1]

        # Keep if it's a local peak or trough
        is_peak = curr_val >= prev_val and curr_val >= next_val
        is_trough = curr_val <= prev_val and curr_val <= next_val

        # Keep if deviation from linear interpolation exceeds tolerance
        if len(indices) >= 1:
            expected = prev_val + (next_val - prev_val) * 0.5
            deviation = abs(curr_val - expected)
            significant_change = deviation > tolerance
        else:
            significant_change = False

        if is_peak or is_trough or significant_change:
            indices.append(i)

    indices.append(len(values) - 1)
    return indices
