"""Funscript file parsing and creation utilities.

Funscript format: JSON with an "actions" array of {at: ms, pos: 0-100} objects.
Internally we normalize positions to [0, 1] and work with frame indices.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

log = logging.getLogger(__name__)


def load_funscript(path: str | Path) -> dict:
    """Load a funscript file and return the raw JSON dict."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_actions(data: dict, trim_ratio: float = 0.0) -> list[tuple[int, int]]:
    """Extract (timestamp_ms, position_0_100) pairs from funscript data.

    Args:
        data: Raw funscript JSON.
        trim_ratio: Optional ratio of actions to drop from each end. Defaults to
            0.0 so label generation preserves the full scripted range.
    """
    actions = data.get("actions", [])
    pairs = [(a["at"], a["pos"]) for a in actions]
    pairs.sort(key=lambda x: x[0])

    if trim_ratio > 0.0 and len(pairs) > 2:
        edge_count = int(len(pairs) * trim_ratio)
        if edge_count > 0 and edge_count * 2 < len(pairs):
            pairs = pairs[edge_count:len(pairs) - edge_count]

    return pairs


def _prepare_interpolation_series(
    actions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, deduplicated action timestamps and normalized positions."""
    if not actions:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    timestamps_ms = np.asarray([a[0] for a in actions], dtype=np.float64)
    positions = np.asarray([a[1] for a in actions], dtype=np.float64) / 100.0

    order = np.argsort(timestamps_ms, kind="stable")
    timestamps_ms = timestamps_ms[order]
    positions = positions[order]

    dedup_timestamps: list[float] = []
    dedup_positions: list[float] = []
    for timestamp_ms, position in zip(timestamps_ms, positions):
        if dedup_timestamps and timestamp_ms == dedup_timestamps[-1]:
            dedup_positions[-1] = float(position)
        else:
            dedup_timestamps.append(float(timestamp_ms))
            dedup_positions.append(float(position))

    return (
        np.asarray(dedup_timestamps, dtype=np.float64),
        np.asarray(dedup_positions, dtype=np.float64),
    )


def actions_to_timestamps(
    actions: list[tuple[int, int]],
    timestamps_ms: np.ndarray,
    interpolation: str = "pchip",
) -> np.ndarray:
    """Sample funscript actions at arbitrary timestamps.

    Timestamps outside the scripted range hold the nearest endpoint value rather
    than extrapolating, which avoids long clipped plateaus from spline drift.
    """
    if timestamps_ms.size == 0:
        return np.zeros((0,), dtype=np.float32)

    action_times_ms, positions = _prepare_interpolation_series(actions)
    if action_times_ms.size == 0:
        return np.zeros_like(timestamps_ms, dtype=np.float32)
    if action_times_ms.size == 1:
        return np.full(timestamps_ms.shape, positions[0], dtype=np.float32)

    clipped_times_ms = np.clip(timestamps_ms.astype(np.float64), action_times_ms[0], action_times_ms[-1])
    interp_kind = interpolation.lower()

    if interp_kind in {"cubic", "pchip", "monotone", "monotonic"}:
        try:
            labels = PchipInterpolator(action_times_ms, positions)(clipped_times_ms)
        except Exception:
            log.warning("Monotone interpolation failed, falling back to linear interpolation")
            interp_kind = "linear"

    if interp_kind == "nearest":
        labels = interp1d(
            action_times_ms,
            positions,
            kind="nearest",
            assume_sorted=True,
        )(clipped_times_ms)
    elif interp_kind == "linear":
        labels = interp1d(
            action_times_ms,
            positions,
            kind="linear",
            assume_sorted=True,
        )(clipped_times_ms)

    labels = np.clip(np.asarray(labels, dtype=np.float64), 0.0, 1.0)
    return labels.astype(np.float32)


def actions_to_frame_labels(
    actions: list[tuple[int, int]],
    fps: float,
    total_frames: int,
    interpolation: str = "pchip",
) -> np.ndarray:
    """Convert funscript actions to per-frame position labels in [0, 1].

    Args:
        actions: List of (timestamp_ms, position_0_100) pairs.
        fps: Video frame rate.
        total_frames: Total number of frames in the video.
        interpolation: "linear", "pchip"/"cubic", or "nearest".

    Returns:
        np.ndarray of shape [total_frames] with values in [0, 1].
    """
    if not actions or fps <= 0 or total_frames <= 0:
        return np.zeros(total_frames, dtype=np.float32)

    frame_timestamps_ms = np.arange(total_frames, dtype=np.float64) * (1000.0 / fps)
    return actions_to_timestamps(actions, frame_timestamps_ms, interpolation=interpolation)


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
