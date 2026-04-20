"""Utilities for DispositionTCN spatial feature extraction and storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


SPATIAL_FORMAT_VERSION = 2
SPATIAL_METHOD = "multiscale_roi_align"
DEFAULT_SCALE_NAMES = ("p3", "p4", "p5")
DEFAULT_SCALE_STRIDES = (8, 16, 32)


def resolve_disposition_feature_layers(
    model: Any,
    layer_indices: Sequence[int] | None = None,
    strides: Sequence[int] | None = None,
    scale_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve the neck feature maps that feed the final detect/pose head.

    By default this inspects the last Detect/Pose-like module and uses its
    input indices. On the current YOLO26 pose models this resolves to the
    P3/P4/P5 feature maps immediately before the head.
    """
    layers = model.model.model

    if layer_indices is None:
        resolved_indices: tuple[int, ...] | None = None
        for idx in range(len(layers) - 1, -1, -1):
            layer = layers[idx]
            from_idx = getattr(layer, "f", None)
            layer_name = type(layer).__name__
            if (
                isinstance(from_idx, (list, tuple))
                and len(from_idx) >= 3
                and all(isinstance(value, int) for value in from_idx)
                and ("Pose" in layer_name or "Detect" in layer_name)
            ):
                resolved_indices = tuple(int(value) for value in from_idx[:3])
                break

        if resolved_indices is None:
            raise ValueError(
                "Could not resolve multiscale feature layers from the final YOLO head"
            )
        layer_indices = resolved_indices
    else:
        layer_indices = tuple(int(value) for value in layer_indices)

    if strides is None:
        if len(layer_indices) == len(DEFAULT_SCALE_STRIDES):
            strides = DEFAULT_SCALE_STRIDES
        else:
            raise ValueError(
                "Explicit strides are required when resolving a non-standard number of feature layers"
            )
    else:
        strides = tuple(int(value) for value in strides)

    if scale_names is None:
        if len(layer_indices) <= len(DEFAULT_SCALE_NAMES):
            scale_names = DEFAULT_SCALE_NAMES[: len(layer_indices)]
        else:
            scale_names = tuple(f"p{i + 3}" for i in range(len(layer_indices)))
    else:
        scale_names = tuple(str(value) for value in scale_names)

    if not (len(layer_indices) == len(strides) == len(scale_names)):
        raise ValueError(
            "layer_indices, strides, and scale_names must have the same length"
        )

    specs: list[dict[str, Any]] = []
    for name, layer_idx, stride in zip(scale_names, layer_indices, strides):
        if not 0 <= layer_idx < len(layers):
            raise ValueError(
                f"Invalid feature layer index {layer_idx} for model with {len(layers)} layers"
            )
        specs.append(
            {
                "name": name,
                "layer_idx": int(layer_idx),
                "layer_name": type(layers[layer_idx]).__name__,
                "stride": int(stride),
            }
        )
    return specs


def build_channel_slices(
    scale_specs: Sequence[dict[str, Any]],
    channel_counts: Sequence[int],
) -> dict[str, list[int]]:
    """Build channel offsets for concatenated multi-scale features."""
    if len(scale_specs) != len(channel_counts):
        raise ValueError("scale_specs and channel_counts must have the same length")

    channel_slices: dict[str, list[int]] = {}
    start = 0
    for spec, channels in zip(scale_specs, channel_counts):
        end = start + int(channels)
        channel_slices[str(spec["name"])] = [start, end]
        start = end
    return channel_slices


def spatial_feature_path(scene_dir: str | Path, model_name: str) -> Path:
    """Return the spatial HDF5 cache path for a scene/model pair."""
    return Path(scene_dir) / "spatial" / f"{model_name}.h5"


def legacy_conf_path(scene_dir: str | Path, model_name: str) -> Path:
    """Return the legacy separate confidence HDF5 path."""
    return Path(scene_dir) / "spatial" / f"{model_name}_conf.h5"


def quantize_spatial_per_channel(spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel int8 quantisation for [T, N, C, H, W] arrays."""
    spatial_fp32 = np.asarray(spatial, dtype=np.float32)
    channel_scale = np.max(np.abs(spatial_fp32), axis=(0, 1, 3, 4)).astype(np.float32)
    channel_scale = np.maximum(channel_scale, 1e-8)
    quantised = np.rint(
        spatial_fp32 / channel_scale.reshape(1, 1, -1, 1, 1)
    ).clip(-127, 127).astype(np.int8)
    return quantised, channel_scale


def dequantize_spatial_per_channel(
    quantised: np.ndarray,
    channel_scale: np.ndarray,
    out_dtype: np.dtype = np.float16,
) -> np.ndarray:
    """Dequantise symmetric per-channel int8 features."""
    spatial = quantised.astype(np.float32) * channel_scale.reshape(1, 1, -1, 1, 1)
    return spatial.astype(out_dtype)


def _json_attr(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def read_spatial_metadata(
    spatial_path: str | Path,
    file_handle: h5py.File | None = None,
) -> dict[str, Any]:
    """Read spatial feature metadata from an HDF5 file."""
    should_close = file_handle is None
    handle = file_handle or h5py.File(str(spatial_path), "r")
    try:
        attrs = handle.attrs
        return {
            "format_version": int(attrs.get("format_version", 1)),
            "method": _maybe_parse_json(attrs.get("method", "single_scale_roi_align")),
            "storage_dtype": _maybe_parse_json(attrs.get("storage_dtype", str(handle["spatial"].dtype))),
            "roi_size": _maybe_parse_json(attrs.get("roi_size", handle["spatial"].shape[-1])),
            "scale_specs": _maybe_parse_json(attrs.get("scale_specs", None)),
            "channel_slices": _maybe_parse_json(attrs.get("channel_slices", None)),
            "source_layers": _maybe_parse_json(attrs.get("source_layers", None)),
            "source_strides": _maybe_parse_json(attrs.get("source_strides", None)),
            "shape": list(handle["spatial"].shape),
        }
    finally:
        if should_close:
            handle.close()


def save_spatial_features_h5(
    spatial_path: str | Path,
    spatial: np.ndarray,
    conf: np.ndarray,
    *,
    storage_dtype: str = "int8",
    compression: str | None = "lzf",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist spatial features and confidence values in a single HDF5 file."""
    out_path = Path(spatial_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spatial_np = np.asarray(spatial)
    conf_np = np.asarray(conf, dtype=np.float32)
    if spatial_np.ndim != 5:
        raise ValueError(f"Expected spatial ndim=5, got {spatial_np.ndim}")
    if conf_np.ndim != 2:
        raise ValueError(f"Expected conf ndim=2, got {conf_np.ndim}")
    if conf_np.shape[:2] != spatial_np.shape[:2]:
        raise ValueError(
            f"Spatial/conf frame-person dimensions must match, got {spatial_np.shape[:2]} and {conf_np.shape[:2]}"
        )

    storage_dtype = storage_dtype.lower()
    compression = None if compression in (None, "", "none") else compression

    if storage_dtype == "int8":
        payload, channel_scale = quantize_spatial_per_channel(spatial_np)
        storage_label = "int8_qchannel"
    elif storage_dtype == "float16":
        payload = np.asarray(spatial_np, dtype=np.float16)
        channel_scale = None
        storage_label = "float16"
    else:
        raise ValueError(f"Unsupported storage_dtype: {storage_dtype}")

    chunk_t = max(1, min(int(payload.shape[0]), 128))
    chunk_spatial = (chunk_t,) + tuple(int(dim) for dim in payload.shape[1:])
    chunk_conf = (chunk_t,) + tuple(int(dim) for dim in conf_np.shape[1:])

    with h5py.File(str(out_path), "w") as handle:
        handle.create_dataset(
            "spatial",
            data=payload,
            dtype=payload.dtype,
            chunks=chunk_spatial,
            compression=compression,
            shuffle=bool(compression),
        )
        handle.create_dataset(
            "conf",
            data=conf_np,
            dtype=np.float32,
            chunks=chunk_conf,
            compression=compression,
            shuffle=bool(compression),
        )
        if channel_scale is not None:
            handle.create_dataset("channel_scale", data=channel_scale, dtype=np.float32)

        attrs = handle.attrs
        attrs["format_version"] = SPATIAL_FORMAT_VERSION
        attrs["method"] = SPATIAL_METHOD
        attrs["storage_dtype"] = storage_label
        attrs["roi_size"] = int(spatial_np.shape[-1])

        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float, np.integer, np.floating)):
                attrs[key] = value
            elif isinstance(value, str):
                attrs[key] = value
            else:
                attrs[key] = _json_attr(value)


def read_spatial_features_h5(
    spatial_path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
    file_handle: h5py.File | None = None,
    legacy_conf_path: str | Path | None = None,
    legacy_conf_file_handle: h5py.File | None = None,
    out_dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a spatial feature slice with backward compatibility for legacy conf files."""
    should_close_spatial = file_handle is None
    spatial_handle = file_handle or h5py.File(str(spatial_path), "r")
    try:
        sl = slice(start, end)
        spatial_raw = spatial_handle["spatial"][sl]
        metadata = read_spatial_metadata(spatial_path, file_handle=spatial_handle)
        storage_dtype = str(metadata.get("storage_dtype", spatial_raw.dtype))

        if spatial_raw.dtype == np.int8 or storage_dtype == "int8_qchannel":
            if "channel_scale" not in spatial_handle:
                raise ValueError(
                    f"Quantised spatial file {spatial_path} is missing channel_scale"
                )
            channel_scale = np.asarray(spatial_handle["channel_scale"][:], dtype=np.float32)
            spatial = dequantize_spatial_per_channel(spatial_raw, channel_scale, out_dtype=out_dtype)
        else:
            spatial = np.asarray(spatial_raw, dtype=out_dtype)

        if "conf" in spatial_handle:
            conf = np.asarray(spatial_handle["conf"][sl], dtype=np.float32)
        else:
            if legacy_conf_file_handle is not None:
                conf = np.asarray(legacy_conf_file_handle["conf"][sl], dtype=np.float32)
            elif legacy_conf_path is not None and Path(legacy_conf_path).exists():
                with h5py.File(str(legacy_conf_path), "r") as legacy_handle:
                    conf = np.asarray(legacy_handle["conf"][sl], dtype=np.float32)
            else:
                raise FileNotFoundError(
                    f"No confidence dataset found in {spatial_path} and no legacy conf path was available"
                )

        return spatial, conf, metadata
    finally:
        if should_close_spatial:
            spatial_handle.close()