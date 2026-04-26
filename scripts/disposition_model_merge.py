"""Compare, merge, and benchmark DispositionTCN checkpoints.

Usage:
    python scripts/disposition_model_merge.py compare \
        --model-a data/models/checkpoints_disposition/best_disposition.pt \
        --model-b data/models/checkpoints_disposition/best_disposition_1e5.pt

    python scripts/disposition_model_merge.py merge \
        --model-a data/models/checkpoints_disposition/best_disposition.pt \
        --model-b data/models/checkpoints_disposition/best_disposition_1e5.pt \
        --include tcn_blocks.0 tcn_blocks.1 \
        --weight-a 0.7 --weight-b 0.3

    python scripts/disposition_model_merge.py gui \
        --model-a data/models/checkpoints_disposition/best_disposition.pt \
        --model-b data/models/checkpoints_disposition/best_disposition_1e5.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_disposition import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    SpatialDataset,
    _load_model_state,
    _prepare_batch,
)
from src.models.dispositiontcn import DispositionTCN, extract_disposition_config  # noqa: E402
from src.training.funscript_metrics import compute_regression_metrics  # noqa: E402


MODEL_TYPE = "disposition_tcn"
DEFAULT_OUTPUT_ROOT = Path("tmp/disposition_model_merges")
DEFAULT_BENCHMARK_COUNT = 10
DEFAULT_BENCHMARK_SEED = 42
DEFAULT_COMPONENT_PLOT_LIMIT = 16
STRUCTURAL_CONFIG_KEYS = (
    "in_channels",
    "roi_size",
    "d_model",
    "n_blocks",
    "kernel_size",
    "encoder_dim",
    "use_ddl",
    "use_aux_layers",
    "scale_channel_slices",
    "scale_names",
)
RUNTIME_CONFIG_KEYS = (
    "dropout",
    "n_persons",
    "output_activation",
)
BENCHMARK_METRIC_KEYS = (
    "pos_mse",
    "event_mse",
    "active_mse",
    "vel_mse",
    "vel_mae",
    "acc_mae",
    "spec_mse",
    "pred_mean",
    "pred_std",
)


log = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckpointBundle:
    label: str
    path: Path
    checkpoint: dict[str, Any]
    model_config: dict[str, Any]
    data_config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]

    @classmethod
    def load(cls, label: str, path: Path) -> "CheckpointBundle":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(f"Checkpoint {path} does not contain a model_config dict")
        model_type = model_config.get("model_type")
        if model_type not in (None, MODEL_TYPE):
            raise ValueError(
                f"Checkpoint {path} is not a disposition checkpoint "
                f"(model_type={model_type!r})"
            )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint {path} does not contain model_state_dict")
        data_config = checkpoint.get("data_config")
        if not isinstance(data_config, dict):
            data_config = {}
        return cls(
            label=label,
            path=path,
            checkpoint=checkpoint,
            model_config=model_config,
            data_config=data_config,
            state_dict=state_dict,
        )

    @property
    def display_name(self) -> str:
        return self.path.name

    def make_model(self) -> DispositionTCN:
        model = DispositionTCN(**extract_disposition_config(self.model_config))
        _load_model_state(model, self.state_dict)
        model.eval()
        return model


@dataclass(slots=True)
class KeyComparison:
    key: str
    status: str
    numel: int = 0
    shape_a: tuple[int, ...] | None = None
    shape_b: tuple[int, ...] | None = None
    dtype_a: str | None = None
    dtype_b: str | None = None
    mean_a: float | None = None
    std_a: float | None = None
    mean_b: float | None = None
    std_b: float | None = None
    mean_sq_a: float | None = None
    mean_sq_b: float | None = None
    mean_abs_diff: float | None = None
    mean_sq_diff: float | None = None
    rms_diff: float | None = None
    max_abs_diff: float | None = None


@dataclass(slots=True)
class NodeComparison:
    path: str
    name: str
    depth: int
    keys: tuple[str, ...]
    key_count: int
    compatible_key_count: int
    missing_in_a: tuple[str, ...]
    missing_in_b: tuple[str, ...]
    mismatched: tuple[str, ...]
    status: str
    param_count: int = 0
    mean_a: float | None = None
    std_a: float | None = None
    mean_b: float | None = None
    std_b: float | None = None
    mean_abs_diff: float | None = None
    rms_diff: float | None = None
    max_abs_diff: float | None = None

    @property
    def mergeable(self) -> bool:
        return self.status == "compatible" and self.compatible_key_count > 0


@dataclass(slots=True)
class ComparisonReport:
    bundle_a: CheckpointBundle
    bundle_b: CheckpointBundle
    key_stats: dict[str, KeyComparison]
    node_stats: dict[str, NodeComparison]
    children: dict[str, tuple[str, ...]]
    structural_config_diffs: dict[str, tuple[Any, Any]]
    runtime_config_diffs: dict[str, tuple[Any, Any]]

    def node(self, path: str) -> NodeComparison:
        normalized = normalize_component_path(path)
        if normalized not in self.node_stats:
            raise KeyError(f"Unknown component path: {path!r}")
        return self.node_stats[normalized]


@dataclass(slots=True)
class BenchmarkSequenceMeta:
    dataset_index: int
    scene_id: str
    start: int


@dataclass(slots=True)
class SequenceBenchmarkResult:
    meta: BenchmarkSequenceMeta
    metrics: dict[str, float]
    prediction: np.ndarray
    target: np.ndarray


@dataclass(slots=True)
class ModelBenchmarkResult:
    label: str
    checkpoint_path: Path
    summary: dict[str, float]
    sequences: list[SequenceBenchmarkResult]
    inference_seconds: float


@dataclass(slots=True)
class BenchmarkConfig:
    data_dir: Path
    split: str
    seq_len: int
    stride: int
    model_name: str
    count: int
    seed: int
    device_dequantize: bool
    indices: tuple[int, ...] | None = None


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    sanitized = sanitized.strip("-._")
    return sanitized or f"merge_{int(time.time())}"


def normalize_component_path(path: str | None) -> str:
    if path is None:
        return ""
    text = str(path).strip().strip("/")
    if text.lower() in {"", "model", "root", "all"}:
        return ""
    return text.rstrip(".")


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.{digits}f}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def save_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _normalize_weights(weight_a: float, weight_b: float) -> tuple[float, float]:
    total = weight_a + weight_b
    if total <= 0:
        raise ValueError("weight_a + weight_b must be greater than zero")
    return weight_a / total, weight_b / total


def _compare_tensor_pair(key: str, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> KeyComparison:
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        return KeyComparison(
            key=key,
            status="shape_mismatch",
            shape_a=tuple(tensor_a.shape),
            shape_b=tuple(tensor_b.shape),
            dtype_a=str(tensor_a.dtype),
            dtype_b=str(tensor_b.dtype),
        )
    if tensor_a.dtype != tensor_b.dtype:
        return KeyComparison(
            key=key,
            status="dtype_mismatch",
            shape_a=tuple(tensor_a.shape),
            shape_b=tuple(tensor_b.shape),
            dtype_a=str(tensor_a.dtype),
            dtype_b=str(tensor_b.dtype),
        )

    flat_a = tensor_a.detach().cpu().float().reshape(-1)
    flat_b = tensor_b.detach().cpu().float().reshape(-1)
    if flat_a.numel() == 0:
        return KeyComparison(
            key=key,
            status="compatible",
            numel=0,
            shape_a=tuple(tensor_a.shape),
            shape_b=tuple(tensor_b.shape),
            dtype_a=str(tensor_a.dtype),
            dtype_b=str(tensor_b.dtype),
            mean_a=0.0,
            std_a=0.0,
            mean_b=0.0,
            std_b=0.0,
            mean_sq_a=0.0,
            mean_sq_b=0.0,
            mean_abs_diff=0.0,
            mean_sq_diff=0.0,
            rms_diff=0.0,
            max_abs_diff=0.0,
        )

    mean_a = float(flat_a.mean().item())
    mean_b = float(flat_b.mean().item())
    mean_sq_a = float((flat_a * flat_a).mean().item())
    mean_sq_b = float((flat_b * flat_b).mean().item())
    diff = flat_b - flat_a
    mean_abs_diff = float(diff.abs().mean().item())
    mean_sq_diff = float((diff * diff).mean().item())
    return KeyComparison(
        key=key,
        status="compatible",
        numel=int(flat_a.numel()),
        shape_a=tuple(tensor_a.shape),
        shape_b=tuple(tensor_b.shape),
        dtype_a=str(tensor_a.dtype),
        dtype_b=str(tensor_b.dtype),
        mean_a=mean_a,
        std_a=math.sqrt(max(mean_sq_a - mean_a * mean_a, 0.0)),
        mean_b=mean_b,
        std_b=math.sqrt(max(mean_sq_b - mean_b * mean_b, 0.0)),
        mean_sq_a=mean_sq_a,
        mean_sq_b=mean_sq_b,
        mean_abs_diff=mean_abs_diff,
        mean_sq_diff=mean_sq_diff,
        rms_diff=math.sqrt(max(mean_sq_diff, 0.0)),
        max_abs_diff=float(diff.abs().max().item()),
    )


def _config_differences(
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    keys: Sequence[str],
) -> dict[str, tuple[Any, Any]]:
    diffs: dict[str, tuple[Any, Any]] = {}
    for key in keys:
        value_a = config_a.get(key)
        value_b = config_b.get(key)
        if value_a != value_b:
            diffs[key] = (value_a, value_b)
    return diffs


def build_comparison_report(bundle_a: CheckpointBundle, bundle_b: CheckpointBundle) -> ComparisonReport:
    key_stats: dict[str, KeyComparison] = {}
    node_keys: dict[str, set[str]] = {"": set()}
    children: dict[str, set[str]] = {"": set()}

    all_keys = sorted(set(bundle_a.state_dict) | set(bundle_b.state_dict), key=natural_sort_key)
    for key in all_keys:
        tensor_a = bundle_a.state_dict.get(key)
        tensor_b = bundle_b.state_dict.get(key)
        if tensor_a is None:
            key_stats[key] = KeyComparison(
                key=key,
                status="missing_in_a",
                shape_b=tuple(tensor_b.shape),
                dtype_b=str(tensor_b.dtype),
            )
        elif tensor_b is None:
            key_stats[key] = KeyComparison(
                key=key,
                status="missing_in_b",
                shape_a=tuple(tensor_a.shape),
                dtype_a=str(tensor_a.dtype),
            )
        else:
            key_stats[key] = _compare_tensor_pair(key, tensor_a, tensor_b)

        node_keys[""] .add(key)
        parent = ""
        prefix_parts: list[str] = []
        for part in key.split("."):
            prefix_parts.append(part)
            prefix = ".".join(prefix_parts)
            node_keys.setdefault(prefix, set()).add(key)
            children.setdefault(parent, set()).add(prefix)
            children.setdefault(prefix, set())
            parent = prefix

    node_stats: dict[str, NodeComparison] = {}
    for path, descendant_keys in node_keys.items():
        sorted_keys = tuple(sorted(descendant_keys, key=natural_sort_key))
        comparisons = [key_stats[key] for key in sorted_keys]
        compatible = [item for item in comparisons if item.status == "compatible"]
        missing_in_a = tuple(item.key for item in comparisons if item.status == "missing_in_a")
        missing_in_b = tuple(item.key for item in comparisons if item.status == "missing_in_b")
        mismatched = tuple(
            item.key
            for item in comparisons
            if item.status not in {"compatible", "missing_in_a", "missing_in_b"}
        )

        status = "incompatible"
        if compatible and not missing_in_a and not missing_in_b and not mismatched:
            status = "compatible"
        elif compatible:
            status = "partial"
        elif missing_in_a or missing_in_b or mismatched:
            status = "incompatible"

        param_count = sum(item.numel for item in compatible)
        mean_a = std_a = mean_b = std_b = mean_abs_diff = rms_diff = max_abs_diff = None
        if param_count > 0:
            mean_a_num = sum((item.mean_a or 0.0) * item.numel for item in compatible)
            mean_b_num = sum((item.mean_b or 0.0) * item.numel for item in compatible)
            mean_sq_a_num = sum((item.mean_sq_a or 0.0) * item.numel for item in compatible)
            mean_sq_b_num = sum((item.mean_sq_b or 0.0) * item.numel for item in compatible)
            mean_abs_diff_num = sum((item.mean_abs_diff or 0.0) * item.numel for item in compatible)
            mean_sq_diff_num = sum((item.mean_sq_diff or 0.0) * item.numel for item in compatible)
            mean_a = mean_a_num / param_count
            mean_b = mean_b_num / param_count
            mean_sq_a = mean_sq_a_num / param_count
            mean_sq_b = mean_sq_b_num / param_count
            mean_abs_diff = mean_abs_diff_num / param_count
            rms_diff = math.sqrt(max(mean_sq_diff_num / param_count, 0.0))
            std_a = math.sqrt(max(mean_sq_a - mean_a * mean_a, 0.0))
            std_b = math.sqrt(max(mean_sq_b - mean_b * mean_b, 0.0))
            max_abs_diff = max((item.max_abs_diff or 0.0) for item in compatible)

        node_stats[path] = NodeComparison(
            path=path,
            name=path.split(".")[-1] if path else "model",
            depth=0 if not path else path.count(".") + 1,
            keys=sorted_keys,
            key_count=len(sorted_keys),
            compatible_key_count=len(compatible),
            missing_in_a=missing_in_a,
            missing_in_b=missing_in_b,
            mismatched=mismatched,
            status=status,
            param_count=param_count,
            mean_a=mean_a,
            std_a=std_a,
            mean_b=mean_b,
            std_b=std_b,
            mean_abs_diff=mean_abs_diff,
            rms_diff=rms_diff,
            max_abs_diff=max_abs_diff,
        )

    normalized_children = {
        parent: tuple(sorted(child_paths, key=natural_sort_key))
        for parent, child_paths in children.items()
    }

    return ComparisonReport(
        bundle_a=bundle_a,
        bundle_b=bundle_b,
        key_stats=key_stats,
        node_stats=node_stats,
        children=normalized_children,
        structural_config_diffs=_config_differences(
            bundle_a.model_config,
            bundle_b.model_config,
            STRUCTURAL_CONFIG_KEYS,
        ),
        runtime_config_diffs=_config_differences(
            bundle_a.model_config,
            bundle_b.model_config,
            RUNTIME_CONFIG_KEYS,
        ),
    )


def top_component_nodes(report: ComparisonReport, limit: int = DEFAULT_COMPONENT_PLOT_LIMIT) -> list[NodeComparison]:
    nodes = [
        node
        for node in report.node_stats.values()
        if node.path
        and node.param_count > 0
        and node.compatible_key_count > 0
        and node.key_count > 1
        and node.depth <= 4
    ]
    nodes.sort(key=lambda item: (item.rms_diff or -1.0, item.param_count), reverse=True)
    return nodes[:limit]


def make_top_components_figure(report: ComparisonReport, limit: int = DEFAULT_COMPONENT_PLOT_LIMIT) -> Figure:
    nodes = top_component_nodes(report, limit=limit)
    if not nodes:
        fig = Figure(figsize=(10, 4))
        ax = fig.subplots(1, 1)
        ax.text(0.5, 0.5, "No compatible components to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [node.path for node in nodes]
    values = [node.rms_diff or 0.0 for node in nodes]
    fig = Figure(figsize=(12, max(4.0, len(nodes) * 0.35)))
    ax = fig.subplots(1, 1)
    ax.barh(range(len(nodes)), values, color="#356e9a")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMS weight delta")
    ax.set_title("Largest compatible component differences")
    fig.tight_layout()
    return fig


def _sample_component_values(
    bundle: CheckpointBundle,
    keys: Sequence[str],
    max_points: int = 20_000,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    if not keys:
        return np.zeros(0, dtype=np.float32)
    points_per_key = max(32, max_points // max(1, len(keys)))
    for key in keys:
        tensor = bundle.state_dict.get(key)
        if tensor is None:
            continue
        flat = tensor.detach().cpu().float().reshape(-1)
        if flat.numel() == 0:
            continue
        if flat.numel() <= points_per_key:
            sampled = flat
        else:
            step = max(1, flat.numel() // points_per_key)
            sampled = flat[::step][:points_per_key]
        samples.append(sampled.numpy())
    if not samples:
        return np.zeros(0, dtype=np.float32)
    values = np.concatenate(samples, axis=0)
    if values.shape[0] > max_points:
        values = values[:max_points]
    return values


def make_component_detail_figure(report: ComparisonReport, component_path: str) -> Figure:
    node = report.node(component_path)
    values_a = _sample_component_values(report.bundle_a, node.keys)
    values_b = _sample_component_values(report.bundle_b, node.keys)
    fig = Figure(figsize=(11, 4.5))
    axes = fig.subplots(1, 2)

    ax_hist, ax_text = axes
    if values_a.size == 0 and values_b.size == 0:
        ax_hist.text(0.5, 0.5, "No compatible values for histogram", ha="center", va="center")
        ax_hist.axis("off")
    else:
        bins = 60
        if values_a.size:
            ax_hist.hist(values_a, bins=bins, alpha=0.55, label=report.bundle_a.label, color="#315f86")
        if values_b.size:
            ax_hist.hist(values_b, bins=bins, alpha=0.55, label=report.bundle_b.label, color="#b45f36")
        ax_hist.set_title(f"Weight distribution: {node.path or 'model'}")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count")
        ax_hist.legend()

    ax_text.axis("off")
    detail_lines = [
        f"component: {node.path or 'model'}",
        f"status: {node.status}",
        f"compatible keys: {node.compatible_key_count}/{node.key_count}",
        f"parameter count: {node.param_count:,}",
        f"{report.bundle_a.label} mean/std: {format_float(node.mean_a)} / {format_float(node.std_a)}",
        f"{report.bundle_b.label} mean/std: {format_float(node.mean_b)} / {format_float(node.std_b)}",
        f"mean abs diff: {format_float(node.mean_abs_diff)}",
        f"rms diff: {format_float(node.rms_diff)}",
        f"max abs diff: {format_float(node.max_abs_diff)}",
    ]
    if node.missing_in_a:
        detail_lines.append(f"missing in {report.bundle_a.label}: {len(node.missing_in_a)}")
    if node.missing_in_b:
        detail_lines.append(f"missing in {report.bundle_b.label}: {len(node.missing_in_b)}")
    if node.mismatched:
        detail_lines.append(f"shape/dtype mismatches: {len(node.mismatched)}")
    ax_text.text(0.0, 1.0, "\n".join(detail_lines), va="top", ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def _node_error_details(node: NodeComparison, report: ComparisonReport) -> str:
    issues: list[str] = []
    if node.missing_in_a:
        issues.append(f"missing in {report.bundle_a.label}: {len(node.missing_in_a)}")
    if node.missing_in_b:
        issues.append(f"missing in {report.bundle_b.label}: {len(node.missing_in_b)}")
    if node.mismatched:
        issues.append(f"shape/dtype mismatches: {len(node.mismatched)}")
    if not issues:
        return node.status
    return "; ".join(issues)


def resolve_selection_keys(report: ComparisonReport, component_paths: Sequence[str] | None) -> tuple[set[str], list[str]]:
    normalized_paths = [normalize_component_path(path) for path in (component_paths or [""])]
    merged_keys: set[str] = set()
    errors: list[str] = []
    for path in normalized_paths:
        if path not in report.node_stats:
            errors.append(f"Unknown component path: {path or 'model'}")
            continue
        node = report.node_stats[path]
        if node.status != "compatible":
            errors.append(f"{path or 'model'} is not fully compatible: {_node_error_details(node, report)}")
            continue
        merged_keys.update(node.keys)
    if not merged_keys and not errors:
        errors.append("No compatible parameters were selected")
    return merged_keys, errors


def merge_tensor_values(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    weight_a: float,
    weight_b: float,
    prefer: str,
) -> torch.Tensor:
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        raise ValueError("Cannot merge tensors with different shapes")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError("Cannot merge tensors with different dtypes")
    if tensor_a.is_floating_point() or tensor_a.is_complex():
        merged = tensor_a.detach().cpu().float() * weight_a + tensor_b.detach().cpu().float() * weight_b
        return merged.to(dtype=tensor_a.dtype)
    if weight_a == weight_b:
        return tensor_a.detach().cpu().clone() if prefer == "a" else tensor_b.detach().cpu().clone()
    return tensor_a.detach().cpu().clone() if weight_a > weight_b else tensor_b.detach().cpu().clone()


def build_merged_checkpoint(
    bundle_a: CheckpointBundle,
    bundle_b: CheckpointBundle,
    report: ComparisonReport,
    component_paths: Sequence[str] | None,
    weight_a: float,
    weight_b: float,
    base_label: str,
) -> tuple[dict[str, Any], set[str]]:
    normalized_weight_a, normalized_weight_b = _normalize_weights(weight_a, weight_b)
    selected_keys, errors = resolve_selection_keys(report, component_paths)
    if errors:
        raise ValueError("; ".join(errors))

    base_bundle = bundle_a if base_label == "a" else bundle_b
    dominant = base_label
    merged_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in base_bundle.state_dict.items()
    }
    for key in selected_keys:
        tensor_a = bundle_a.state_dict.get(key)
        tensor_b = bundle_b.state_dict.get(key)
        if tensor_a is None or tensor_b is None:
            raise ValueError(f"Selected key {key} is not present in both checkpoints")
        merged_state[key] = merge_tensor_values(
            tensor_a,
            tensor_b,
            normalized_weight_a,
            normalized_weight_b,
            prefer=dominant,
        )

    merged_checkpoint = {
        "epoch": 0,
        "model_state_dict": merged_state,
        "val_loss": None,
        "model_config": copy.deepcopy(base_bundle.model_config),
        "data_config": copy.deepcopy(base_bundle.data_config),
        "merge_metadata": {
            "created_at": int(time.time()),
            "model_a": str(bundle_a.path),
            "model_b": str(bundle_b.path),
            "base_model": base_label,
            "weight_a": normalized_weight_a,
            "weight_b": normalized_weight_b,
            "selected_components": [normalize_component_path(path) for path in (component_paths or [""])],
            "selected_key_count": len(selected_keys),
        },
    }

    validation_model = DispositionTCN(**extract_disposition_config(merged_checkpoint["model_config"]))
    _load_model_state(validation_model, merged_checkpoint["model_state_dict"])
    return merged_checkpoint, selected_keys


def _batchify_item(item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batched: dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batched[key] = value.unsqueeze(0)
    return batched


def resolve_benchmark_config(
    args: argparse.Namespace,
    bundle_a: CheckpointBundle,
) -> BenchmarkConfig:
    device = resolve_device(args.device)
    seq_len = int(args.seq_len or bundle_a.data_config.get("seq_len") or 120)
    stride = int(args.stride or seq_len)
    model_name = str(args.model_name or bundle_a.data_config.get("model_name") or DEFAULT_MODEL_NAME)
    if args.device_dequantize is None:
        device_dequantize = device.type == "cuda" and bool(bundle_a.data_config.get("device_dequantize", True))
    else:
        device_dequantize = bool(args.device_dequantize)
    explicit_indices = tuple(args.benchmark_indices) if args.benchmark_indices else None
    return BenchmarkConfig(
        data_dir=args.data_dir,
        split=args.benchmark_split,
        seq_len=seq_len,
        stride=stride,
        model_name=model_name,
        count=args.benchmark_count,
        seed=args.benchmark_seed,
        device_dequantize=device_dequantize,
        indices=explicit_indices,
    )


def select_benchmark_indices(total: int, count: int, seed: int, explicit: tuple[int, ...] | None) -> list[int]:
    if explicit is not None:
        indices = [int(idx) for idx in explicit]
        if not indices:
            raise ValueError("benchmark_indices cannot be empty")
        for idx in indices:
            if idx < 0 or idx >= total:
                raise ValueError(f"Benchmark index {idx} is out of range for dataset size {total}")
        return indices
    if total <= 0:
        raise ValueError("Benchmark dataset is empty")
    count = min(total, max(1, count))
    rng = np.random.default_rng(seed)
    indices = sorted(int(idx) for idx in rng.choice(total, size=count, replace=False))
    return indices


def split_main_prediction(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim == 2:
        return prediction
    if prediction.ndim == 3:
        return prediction[:, 0]
    raise ValueError(f"Unsupported model output shape {tuple(prediction.shape)}")


def run_benchmark_suite(
    model_entries: Sequence[tuple[str, Path, DispositionTCN]],
    benchmark_config: BenchmarkConfig,
    device: torch.device,
) -> tuple[list[BenchmarkSequenceMeta], dict[str, ModelBenchmarkResult]]:
    dataset = SpatialDataset(
        benchmark_config.data_dir,
        benchmark_config.split,
        seq_len=benchmark_config.seq_len,
        stride=benchmark_config.stride,
        model_name=benchmark_config.model_name,
        augment=False,
        device_dequantize=benchmark_config.device_dequantize,
    )
    try:
        # Helpful failure when no benchmark sequences are available so GUI shows
        # a clear actionable message instead of a generic traceback.
        if len(dataset) == 0:
            raise ValueError(
                f"Benchmark dataset is empty for split={benchmark_config.split!r} model_name={benchmark_config.model_name!r} "
                f"under data_dir={benchmark_config.data_dir}. Ensure spatial caches and labels exist for the chosen model_name/ split, or adjust --model-name / --benchmark-split."
            )
        indices = select_benchmark_indices(
            len(dataset),
            benchmark_config.count,
            benchmark_config.seed,
            benchmark_config.indices,
        )
        sequence_meta = [
            BenchmarkSequenceMeta(
                dataset_index=idx,
                scene_id=dataset.sequences[idx][0],
                start=dataset.sequences[idx][1],
            )
            for idx in indices
        ]
        raw_batches = [_batchify_item(dataset[idx]) for idx in indices]
        results: dict[str, ModelBenchmarkResult] = {}

        for label, checkpoint_path, model in model_entries:
            model = model.to(device)
            model.eval()
            per_sequence: list[SequenceBenchmarkResult] = []
            start_time = time.perf_counter()
            with torch.inference_mode():
                for meta, raw_batch in zip(sequence_meta, raw_batches):
                    prepared = _prepare_batch(
                        raw_batch,
                        device,
                        device_dequantize=dataset.device_dequantize,
                    )
                    spatial = prepared["spatial"]
                    conf = prepared["conf"]
                    labels = prepared["labels"]
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        raw_prediction = model(spatial, conf)
                    prediction = split_main_prediction(raw_prediction).float()
                    labels = labels.float()
                    metrics = compute_regression_metrics(prediction, labels, spectral_kernel=15)
                    metric_values = {key: float(metrics[key].item()) for key in metrics}
                    prediction_np = prediction[0].float().cpu().numpy()
                    target_np = labels[0].float().cpu().numpy()
                    metric_values["pred_mean"] = float(prediction_np.mean())
                    metric_values["pred_std"] = float(prediction_np.std())
                    per_sequence.append(
                        SequenceBenchmarkResult(
                            meta=meta,
                            metrics=metric_values,
                            prediction=prediction_np,
                            target=target_np,
                        )
                    )
            inference_seconds = time.perf_counter() - start_time

            summary: dict[str, float] = {
                "sequence_count": float(len(per_sequence)),
                "inference_seconds": float(inference_seconds),
                "seconds_per_sequence": float(inference_seconds / max(1, len(per_sequence))),
            }
            for key in BENCHMARK_METRIC_KEYS:
                values = [item.metrics[key] for item in per_sequence if key in item.metrics]
                if not values:
                    continue
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))

            results[label] = ModelBenchmarkResult(
                label=label,
                checkpoint_path=checkpoint_path,
                summary=summary,
                sequences=per_sequence,
                inference_seconds=float(inference_seconds),
            )
    finally:
        dataset.close()

    return sequence_meta, results


def make_benchmark_summary_figure(results: dict[str, ModelBenchmarkResult]) -> Figure:
    labels = list(results)
    fig = Figure(figsize=(12, 7))
    axes = fig.subplots(2, 2)
    metric_specs = [
        ("pos_mse_mean", "Position MSE"),
        ("vel_mae_mean", "Velocity MAE"),
        ("active_mse_mean", "Active MSE"),
        ("seconds_per_sequence", "Seconds / sequence"),
    ]
    for ax, (metric_key, title) in zip(axes.reshape(-1), metric_specs):
        values = [results[label].summary.get(metric_key, float("nan")) for label in labels]
        ax.bar(labels, values, color=["#315f86", "#b45f36", "#5b8f3c"][: len(labels)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def make_sequence_overlay_figure(
    sequence_meta: Sequence[BenchmarkSequenceMeta],
    results: dict[str, ModelBenchmarkResult],
    sequence_index: int = 0,
) -> Figure:
    if not sequence_meta:
        fig = Figure(figsize=(10, 4))
        ax = fig.subplots(1, 1)
        ax.text(0.5, 0.5, "No benchmark sequences available", ha="center", va="center")
        ax.axis("off")
        return fig

    meta = sequence_meta[sequence_index]
    fig = Figure(figsize=(12, 4.5))
    ax = fig.subplots(1, 1)
    target = next(iter(results.values())).sequences[sequence_index].target
    ax.plot(target, color="#111111", lw=2.0, label="target")
    colors = ["#315f86", "#b45f36", "#5b8f3c"]
    for color, (label, result) in zip(colors, results.items()):
        ax.plot(result.sequences[sequence_index].prediction, color=color, lw=1.5, label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Benchmark overlay: {meta.scene_id} @ {meta.start}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Prediction")
    ax.legend()
    fig.tight_layout()
    return fig


def write_comparison_artifacts(output_dir: Path, report: ComparisonReport) -> None:
    ensure_dir(output_dir)
    node_rows = []
    for path, node in sorted(report.node_stats.items(), key=lambda item: natural_sort_key(item[0])):
        node_rows.append({
            "path": path or "model",
            "depth": node.depth,
            "status": node.status,
            "key_count": node.key_count,
            "compatible_key_count": node.compatible_key_count,
            "param_count": node.param_count,
            "mean_a": format_float(node.mean_a),
            "std_a": format_float(node.std_a),
            "mean_b": format_float(node.mean_b),
            "std_b": format_float(node.std_b),
            "mean_abs_diff": format_float(node.mean_abs_diff),
            "rms_diff": format_float(node.rms_diff),
            "max_abs_diff": format_float(node.max_abs_diff),
            "missing_in_a": len(node.missing_in_a),
            "missing_in_b": len(node.missing_in_b),
            "mismatched": len(node.mismatched),
        })
    save_csv(
        output_dir / "component_stats.csv",
        [
            "path",
            "depth",
            "status",
            "key_count",
            "compatible_key_count",
            "param_count",
            "mean_a",
            "std_a",
            "mean_b",
            "std_b",
            "mean_abs_diff",
            "rms_diff",
            "max_abs_diff",
            "missing_in_a",
            "missing_in_b",
            "mismatched",
        ],
        node_rows,
    )

    key_rows = []
    for key, stats in sorted(report.key_stats.items(), key=lambda item: natural_sort_key(item[0])):
        key_rows.append({
            "key": key,
            "status": stats.status,
            "numel": stats.numel,
            "shape_a": list(stats.shape_a) if stats.shape_a is not None else "",
            "shape_b": list(stats.shape_b) if stats.shape_b is not None else "",
            "dtype_a": stats.dtype_a or "",
            "dtype_b": stats.dtype_b or "",
            "mean_a": format_float(stats.mean_a),
            "std_a": format_float(stats.std_a),
            "mean_b": format_float(stats.mean_b),
            "std_b": format_float(stats.std_b),
            "mean_abs_diff": format_float(stats.mean_abs_diff),
            "rms_diff": format_float(stats.rms_diff),
            "max_abs_diff": format_float(stats.max_abs_diff),
        })
    save_csv(
        output_dir / "parameter_stats.csv",
        [
            "key",
            "status",
            "numel",
            "shape_a",
            "shape_b",
            "dtype_a",
            "dtype_b",
            "mean_a",
            "std_a",
            "mean_b",
            "std_b",
            "mean_abs_diff",
            "rms_diff",
            "max_abs_diff",
        ],
        key_rows,
    )

    top_nodes = top_component_nodes(report)
    save_json(
        output_dir / "summary.json",
        {
            "model_a": str(report.bundle_a.path),
            "model_b": str(report.bundle_b.path),
            "structural_config_diffs": report.structural_config_diffs,
            "runtime_config_diffs": report.runtime_config_diffs,
            "root_status": report.node_stats[""].status,
            "top_components": [
                {
                    "path": node.path,
                    "status": node.status,
                    "param_count": node.param_count,
                    "rms_diff": node.rms_diff,
                    "mean_abs_diff": node.mean_abs_diff,
                    "mean_a": node.mean_a,
                    "std_a": node.std_a,
                    "mean_b": node.mean_b,
                    "std_b": node.std_b,
                }
                for node in top_nodes
            ],
        },
    )
    save_figure(make_top_components_figure(report), output_dir / "top_component_differences.png")


def write_model_benchmark_artifacts(output_dir: Path, result: ModelBenchmarkResult) -> None:
    results_dir = ensure_dir(output_dir / "results")
    save_json(results_dir / "summary.json", result.summary)
    rows = []
    for seq_result in result.sequences:
        row = {
            "dataset_index": seq_result.meta.dataset_index,
            "scene_id": seq_result.meta.scene_id,
            "start": seq_result.meta.start,
        }
        row.update(seq_result.metrics)
        rows.append(row)
    fieldnames = ["dataset_index", "scene_id", "start", *sorted({key for row in rows for key in row if key not in {"dataset_index", "scene_id", "start"}})]
    save_csv(results_dir / "per_sequence.csv", fieldnames, rows)


def write_suite_benchmark_artifacts(
    output_dir: Path,
    sequence_meta: Sequence[BenchmarkSequenceMeta],
    results: dict[str, ModelBenchmarkResult],
) -> None:
    ensure_dir(output_dir)
    save_json(
        output_dir / "summary.json",
        {label: result.summary for label, result in results.items()},
    )
    save_figure(make_benchmark_summary_figure(results), output_dir / "benchmark_summary.png")
    save_figure(make_sequence_overlay_figure(sequence_meta, results, sequence_index=0), output_dir / "benchmark_overlay_0.png")
    per_sequence_dir = ensure_dir(output_dir / "sequences")
    for index, meta in enumerate(sequence_meta):
        save_figure(
            make_sequence_overlay_figure(sequence_meta, results, sequence_index=index),
            per_sequence_dir / f"seq_{index:02d}_{sanitize_name(meta.scene_id)}_{meta.start}.png",
        )


def create_run_dir(output_root: Path, run_name: str | None) -> Path:
    output_root = ensure_dir(output_root)
    if run_name:
        run_dir = output_root / sanitize_name(run_name)
        suffix = 1
        while run_dir.exists():
            run_dir = output_root / f"{sanitize_name(run_name)}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    run_dir = output_root / f"merge_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_checkpoint_to_model_dir(bundle: CheckpointBundle, model_dir: Path) -> None:
    checkpoint_dir = ensure_dir(model_dir / "checkpoint")
    shutil.copy2(bundle.path, checkpoint_dir / bundle.path.name)


def save_merged_checkpoint(checkpoint: dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    torch.save(checkpoint, output_path)


def execute_merge_run(
    bundle_a: CheckpointBundle,
    bundle_b: CheckpointBundle,
    report: ComparisonReport,
    args: argparse.Namespace,
) -> tuple[Path, Path, list[BenchmarkSequenceMeta], dict[str, ModelBenchmarkResult]]:
    run_dir = create_run_dir(args.output_root, args.run_name)
    comparison_dir = ensure_dir(run_dir / "comparison")
    write_comparison_artifacts(comparison_dir, report)

    model_a_dir = ensure_dir(run_dir / "model_a")
    model_b_dir = ensure_dir(run_dir / "model_b")
    model_c_dir = ensure_dir(run_dir / "model_c")
    copy_checkpoint_to_model_dir(bundle_a, model_a_dir)
    copy_checkpoint_to_model_dir(bundle_b, model_b_dir)

    merged_checkpoint, selected_keys = build_merged_checkpoint(
        bundle_a,
        bundle_b,
        report,
        component_paths=args.include,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
        base_label=args.base_model,
    )
    merged_checkpoint_path = model_c_dir / "checkpoint" / "merged_disposition.pt"
    save_merged_checkpoint(merged_checkpoint, merged_checkpoint_path)

    benchmark_config = resolve_benchmark_config(args, bundle_a)
    device = resolve_device(args.device)
    sequence_meta: list[BenchmarkSequenceMeta] = []
    benchmark_results: dict[str, ModelBenchmarkResult] = {}
    if not args.no_benchmark:
        model_entries = [
            (bundle_a.label, bundle_a.path, bundle_a.make_model()),
            (bundle_b.label, bundle_b.path, bundle_b.make_model()),
            ("model_c", merged_checkpoint_path, CheckpointBundle.load("model_c", merged_checkpoint_path).make_model()),
        ]
        sequence_meta, benchmark_results = run_benchmark_suite(model_entries, benchmark_config, device)
        write_model_benchmark_artifacts(model_a_dir, benchmark_results[bundle_a.label])
        write_model_benchmark_artifacts(model_b_dir, benchmark_results[bundle_b.label])
        write_model_benchmark_artifacts(model_c_dir, benchmark_results["model_c"])
        write_suite_benchmark_artifacts(run_dir / "benchmark_suite", sequence_meta, benchmark_results)

    save_json(
        run_dir / "manifest.json",
        {
            "model_a": str(bundle_a.path),
            "model_b": str(bundle_b.path),
            "model_c": str(merged_checkpoint_path),
            "selected_components": [normalize_component_path(path) for path in (args.include or [""])],
            "selected_key_count": len(selected_keys),
            "base_model": args.base_model,
            "weight_a": args.weight_a,
            "weight_b": args.weight_b,
            "benchmark": {
                "enabled": not args.no_benchmark,
                "sequence_count": len(sequence_meta),
                "sequence_meta": [asdict(meta) for meta in sequence_meta],
            },
        },
    )
    return run_dir, merged_checkpoint_path, sequence_meta, benchmark_results


def _build_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-a", type=Path, required=True, help="Path to checkpoint A")
    parser.add_argument("--model-b", type=Path, required=True, help="Path to checkpoint B")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)


def _build_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(device_dequantize=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--benchmark-split", choices=("train", "val"), default="val")
    parser.add_argument("--benchmark-count", type=int, default=DEFAULT_BENCHMARK_COUNT)
    parser.add_argument("--benchmark-seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument(
        "--benchmark-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit validation sequence indices to benchmark",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device-dequantize", action="store_true", dest="device_dequantize")
    parser.add_argument("--no-device-dequantize", action="store_false", dest="device_dequantize")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Disposition checkpoint compare and merge tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Write comparison artifacts for two checkpoints")
    _build_common_model_args(compare_parser)

    merge_parser = subparsers.add_parser("merge", help="Merge compatible components and benchmark the result")
    _build_common_model_args(merge_parser)
    _build_common_benchmark_args(merge_parser)
    merge_parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Component prefixes to merge; omit to merge the whole compatible model",
    )
    merge_parser.add_argument("--weight-a", type=float, default=0.5)
    merge_parser.add_argument("--weight-b", type=float, default=0.5)
    merge_parser.add_argument("--base-model", choices=("a", "b"), default="a")
    merge_parser.add_argument("--no-benchmark", action="store_true")

    gui_parser = subparsers.add_parser("gui", help="Launch the interactive compare/merge UI")
    gui_parser.add_argument("--model-a", type=Path, default=None)
    gui_parser.add_argument("--model-b", type=Path, default=None)
    gui_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    gui_parser.add_argument("--device", type=str, default="auto")
    gui_parser.add_argument("--benchmark-split", choices=("train", "val"), default="val")
    gui_parser.add_argument("--benchmark-count", type=int, default=DEFAULT_BENCHMARK_COUNT)
    gui_parser.add_argument("--benchmark-seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    gui_parser.add_argument("--seq-len", type=int, default=None)
    gui_parser.add_argument("--stride", type=int, default=None)
    gui_parser.add_argument("--model-name", type=str, default=None)
    gui_parser.add_argument("--weight-a", type=float, default=0.5)
    gui_parser.add_argument("--weight-b", type=float, default=0.5)
    gui_parser.add_argument("--base-model", choices=("a", "b"), default="a")
    gui_parser.set_defaults(device_dequantize=None)
    gui_parser.add_argument("--device-dequantize", action="store_true", dest="device_dequantize")
    gui_parser.add_argument("--no-device-dequantize", action="store_false", dest="device_dequantize")

    return parser


def command_compare(args: argparse.Namespace) -> int:
    bundle_a = CheckpointBundle.load("model_a", args.model_a)
    bundle_b = CheckpointBundle.load("model_b", args.model_b)
    report = build_comparison_report(bundle_a, bundle_b)
    run_dir = create_run_dir(args.output_root, args.run_name or f"compare_{bundle_a.path.stem}_vs_{bundle_b.path.stem}")
    write_comparison_artifacts(run_dir / "comparison", report)
    copy_checkpoint_to_model_dir(bundle_a, run_dir / "model_a")
    copy_checkpoint_to_model_dir(bundle_b, run_dir / "model_b")
    save_json(
        run_dir / "manifest.json",
        {
            "model_a": str(bundle_a.path),
            "model_b": str(bundle_b.path),
            "root_status": report.node_stats[""].status,
            "structural_config_diffs": report.structural_config_diffs,
            "runtime_config_diffs": report.runtime_config_diffs,
        },
    )
    log.info("Comparison artifacts written to %s", run_dir)
    return 0


def command_merge(args: argparse.Namespace) -> int:
    bundle_a = CheckpointBundle.load("model_a", args.model_a)
    bundle_b = CheckpointBundle.load("model_b", args.model_b)
    report = build_comparison_report(bundle_a, bundle_b)
    run_dir, merged_checkpoint_path, sequence_meta, benchmark_results = execute_merge_run(
        bundle_a,
        bundle_b,
        report,
        args,
    )
    log.info("Merge artifacts written to %s", run_dir)
    log.info("Merged checkpoint: %s", merged_checkpoint_path)
    if sequence_meta:
        for label, result in benchmark_results.items():
            log.info(
                "%s benchmark pos_mse=%.6f vel_mae=%.6f seconds/seq=%.4f",
                label,
                result.summary.get("pos_mse_mean", float("nan")),
                result.summary.get("vel_mae_mean", float("nan")),
                result.summary.get("seconds_per_sequence", float("nan")),
            )
    return 0


def launch_gui(args: argparse.Namespace) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    class MergeGuiApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("Disposition Model Merge")
            self.root.geometry("1600x980")

            self.model_a_var = tk.StringVar(value=str(args.model_a) if args.model_a else "")
            self.model_b_var = tk.StringVar(value=str(args.model_b) if args.model_b else "")
            self.output_root_var = tk.StringVar(value=str(args.output_root))
            self.weight_a_var = tk.StringVar(value=str(args.weight_a))
            self.weight_b_var = tk.StringVar(value=str(args.weight_b))
            self.base_model_var = tk.StringVar(value=args.base_model)
            self.benchmark_count_var = tk.StringVar(value=str(args.benchmark_count))
            self.benchmark_seed_var = tk.StringVar(value=str(args.benchmark_seed))
            self.seq_len_var = tk.StringVar(value="" if args.seq_len is None else str(args.seq_len))
            self.stride_var = tk.StringVar(value="" if args.stride is None else str(args.stride))
            self.model_name_var = tk.StringVar(value=args.model_name or "")
            self.device_var = tk.StringVar(value=args.device)
            self.status_var = tk.StringVar(value="Load two checkpoints to compare and merge.")

            self.bundle_a: CheckpointBundle | None = None
            self.bundle_b: CheckpointBundle | None = None
            self.report: ComparisonReport | None = None
            self.tree_path_map: dict[str, str] = {}
            self.last_run_dir: Path | None = None
            self.last_benchmark_results: dict[str, ModelBenchmarkResult] | None = None
            self.last_sequence_meta: list[BenchmarkSequenceMeta] = []

            self._build_ui()
            if self.model_a_var.get() and self.model_b_var.get():
                self.load_models()

        def _build_ui(self) -> None:
            outer = ttk.Frame(self.root, padding=10)
            outer.pack(fill="both", expand=True)

            load_frame = ttk.LabelFrame(outer, text="Checkpoints", padding=8)
            load_frame.pack(fill="x")
            self._path_row(load_frame, 0, "Model A", self.model_a_var)
            self._path_row(load_frame, 1, "Model B", self.model_b_var)
            ttk.Button(load_frame, text="Load models", command=self.load_models).grid(row=0, column=4, rowspan=2, padx=8, sticky="ns")

            options = ttk.LabelFrame(outer, text="Merge Options", padding=8)
            options.pack(fill="x", pady=(8, 8))
            labels = [
                ("Weight A", self.weight_a_var),
                ("Weight B", self.weight_b_var),
                ("Benchmark count", self.benchmark_count_var),
                ("Benchmark seed", self.benchmark_seed_var),
                ("Seq len", self.seq_len_var),
                ("Stride", self.stride_var),
                ("Model name", self.model_name_var),
                ("Device", self.device_var),
            ]
            for column, (label, variable) in enumerate(labels):
                ttk.Label(options, text=label).grid(row=0, column=column, sticky="w")
                ttk.Entry(options, textvariable=variable, width=16).grid(row=1, column=column, padx=(0, 8), sticky="ew")
            ttk.Label(options, text="Base model").grid(row=0, column=len(labels), sticky="w")
            ttk.Combobox(options, textvariable=self.base_model_var, values=("a", "b"), width=8, state="readonly").grid(row=1, column=len(labels), padx=(0, 8), sticky="ew")
            ttk.Label(options, text="Output root").grid(row=2, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.output_root_var).grid(row=3, column=0, columnspan=8, sticky="ew", pady=(0, 4))
            ttk.Button(options, text="Browse", command=self._browse_output_root).grid(row=3, column=8, sticky="ew", padx=(0, 8))
            ttk.Button(options, text="Select whole model", command=self.select_root).grid(row=3, column=9, sticky="ew", padx=(0, 8))
            ttk.Button(options, text="Select top 5 diffs", command=self.select_top_diffs).grid(row=3, column=10, sticky="ew", padx=(0, 8))
            ttk.Button(options, text="Clear selection", command=self.clear_selection).grid(row=3, column=11, sticky="ew", padx=(0, 8))
            ttk.Button(options, text="Merge selected", command=self.merge_selected).grid(row=3, column=12, sticky="ew")
            for column in range(13):
                options.columnconfigure(column, weight=1)

            split = ttk.Panedwindow(outer, orient="horizontal")
            split.pack(fill="both", expand=True)

            left = ttk.Frame(split)
            right = ttk.Frame(split)
            split.add(left, weight=1)
            split.add(right, weight=2)

            tree_frame = ttk.LabelFrame(left, text="Components", padding=6)
            tree_frame.pack(fill="both", expand=True)
            columns = ("status", "params", "rms_diff", "std_a", "std_b")
            self.tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", selectmode="extended")
            self.tree.heading("#0", text="Component")
            self.tree.heading("status", text="Status")
            self.tree.heading("params", text="Params")
            self.tree.heading("rms_diff", text="RMS diff")
            self.tree.heading("std_a", text="A std")
            self.tree.heading("std_b", text="B std")
            self.tree.column("#0", width=360, stretch=True)
            self.tree.column("status", width=90, anchor="center")
            self.tree.column("params", width=90, anchor="e")
            self.tree.column("rms_diff", width=90, anchor="e")
            self.tree.column("std_a", width=80, anchor="e")
            self.tree.column("std_b", width=80, anchor="e")
            tree_scroll_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
            tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
            self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
            self.tree.grid(row=0, column=0, sticky="nsew")
            tree_scroll_y.grid(row=0, column=1, sticky="ns")
            tree_scroll_x.grid(row=1, column=0, sticky="ew")
            tree_frame.rowconfigure(0, weight=1)
            tree_frame.columnconfigure(0, weight=1)
            self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

            self.notebook = ttk.Notebook(right)
            self.notebook.pack(fill="both", expand=True)
            self.overview_tab = ttk.Frame(self.notebook)
            self.detail_tab = ttk.Frame(self.notebook)
            self.results_tab = ttk.Frame(self.notebook)
            self.notebook.add(self.overview_tab, text="Overview")
            self.notebook.add(self.detail_tab, text="Component Detail")
            self.notebook.add(self.results_tab, text="Benchmark")

            self.overview_text = tk.Text(self.overview_tab, height=10, wrap="word")
            self.overview_text.pack(fill="x", padx=6, pady=6)
            self.overview_canvas = self._make_canvas(self.overview_tab)

            self.detail_text = tk.Text(self.detail_tab, height=10, wrap="word")
            self.detail_text.pack(fill="x", padx=6, pady=6)
            self.detail_canvas = self._make_canvas(self.detail_tab)

            self.results_text = tk.Text(self.results_tab, height=10, wrap="word")
            self.results_text.pack(fill="x", padx=6, pady=6)
            self.results_canvas = self._make_canvas(self.results_tab)

            status = ttk.Label(outer, textvariable=self.status_var, anchor="w")
            status.pack(fill="x", pady=(8, 0))

        def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=4)
            ttk.Button(parent, text="Browse", command=lambda var=variable: self._browse_checkpoint(var)).grid(row=row, column=3, padx=(0, 8), pady=4)
            parent.columnconfigure(1, weight=1)

        def _make_canvas(self, parent: ttk.Frame) -> FigureCanvasTkAgg:
            figure = Figure(figsize=(8, 4))
            canvas = FigureCanvasTkAgg(figure, master=parent)
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
            return canvas

        def _browse_checkpoint(self, variable: tk.StringVar) -> None:
            path = filedialog.askopenfilename(
                title="Select disposition checkpoint",
                filetypes=[("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*.*")],
            )
            if path:
                variable.set(path)

        def _browse_output_root(self) -> None:
            path = filedialog.askdirectory(title="Select output root")
            if path:
                self.output_root_var.set(path)

        def _set_text(self, widget: tk.Text, text: str) -> None:
            widget.delete("1.0", "end")
            widget.insert("1.0", text)

        def _set_canvas_figure(self, canvas: FigureCanvasTkAgg, fig: Figure) -> None:
            canvas.figure = fig
            canvas.draw()

        def load_models(self) -> None:
            try:
                self.bundle_a = CheckpointBundle.load("model_a", Path(self.model_a_var.get()))
                self.bundle_b = CheckpointBundle.load("model_b", Path(self.model_b_var.get()))
                self.report = build_comparison_report(self.bundle_a, self.bundle_b)
            except Exception as exc:
                messagebox.showerror("Load failed", str(exc))
                self.status_var.set(f"Load failed: {exc}")
                return
            self.populate_tree()
            self.update_overview()
            self.status_var.set("Models loaded. Select compatible components and click Merge selected.")

        def populate_tree(self) -> None:
            assert self.report is not None
            self.tree.delete(*self.tree.get_children())
            self.tree_path_map.clear()

            def insert_node(parent_item: str, path: str) -> None:
                node = self.report.node_stats[path]
                label = path.split(".")[-1] if path else "model"
                item_id = self.tree.insert(
                    parent_item,
                    "end",
                    text=label,
                    values=(
                        node.status,
                        f"{node.param_count:,}",
                        format_float(node.rms_diff),
                        format_float(node.std_a),
                        format_float(node.std_b),
                    ),
                    open=path.count(".") < 2,
                )
                self.tree_path_map[item_id] = path
                for child in self.report.children.get(path, ()): 
                    insert_node(item_id, child)

            insert_node("", "")

        def update_overview(self) -> None:
            assert self.report is not None
            root_node = self.report.node_stats[""]
            lines = [
                f"Model A: {self.report.bundle_a.path}",
                f"Model B: {self.report.bundle_b.path}",
                f"Full model status: {root_node.status}",
                f"Structural config diffs: {len(self.report.structural_config_diffs)}",
                f"Runtime config diffs: {len(self.report.runtime_config_diffs)}",
            ]
            if self.report.structural_config_diffs:
                lines.append("")
                lines.append("Structural config differences:")
                for key, (value_a, value_b) in sorted(self.report.structural_config_diffs.items()):
                    lines.append(f"  {key}: A={value_a!r} | B={value_b!r}")
            if self.report.runtime_config_diffs:
                lines.append("")
                lines.append("Runtime config differences:")
                for key, (value_a, value_b) in sorted(self.report.runtime_config_diffs.items()):
                    lines.append(f"  {key}: A={value_a!r} | B={value_b!r}")
            self._set_text(self.overview_text, "\n".join(lines))
            self._set_canvas_figure(self.overview_canvas, make_top_components_figure(self.report))

        def on_tree_select(self, _event: object) -> None:
            if self.report is None:
                return
            selection = self.tree.selection()
            if not selection:
                return
            first_item = selection[0]
            path = self.tree_path_map[first_item]
            node = self.report.node_stats[path]
            lines = [
                f"Component: {path or 'model'}",
                f"Status: {node.status}",
                f"Compatible keys: {node.compatible_key_count}/{node.key_count}",
                f"Parameters: {node.param_count:,}",
                f"A mean/std: {format_float(node.mean_a)} / {format_float(node.std_a)}",
                f"B mean/std: {format_float(node.mean_b)} / {format_float(node.std_b)}",
                f"Mean abs diff: {format_float(node.mean_abs_diff)}",
                f"RMS diff: {format_float(node.rms_diff)}",
                f"Max abs diff: {format_float(node.max_abs_diff)}",
            ]
            if node.status != "compatible":
                lines.append(_node_error_details(node, self.report))
            self._set_text(self.detail_text, "\n".join(lines))
            self._set_canvas_figure(self.detail_canvas, make_component_detail_figure(self.report, path))

        def select_root(self) -> None:
            for item_id, path in self.tree_path_map.items():
                if path == "":
                    self.tree.selection_set(item_id)
                    self.tree.see(item_id)
                    self.on_tree_select(None)
                    break

        def select_top_diffs(self) -> None:
            if self.report is None:
                return
            target_paths = {node.path for node in top_component_nodes(self.report, limit=5)}
            selection = [item_id for item_id, path in self.tree_path_map.items() if path in target_paths]
            if selection:
                self.tree.selection_set(selection)
                self.tree.see(selection[0])
                self.on_tree_select(None)

        def clear_selection(self) -> None:
            self.tree.selection_remove(self.tree.selection())

        def _build_merge_args(self, selected_paths: list[str]) -> argparse.Namespace:
            namespace = argparse.Namespace(
                output_root=Path(self.output_root_var.get()),
                run_name=None,
                include=selected_paths,
                weight_a=float(self.weight_a_var.get()),
                weight_b=float(self.weight_b_var.get()),
                base_model=self.base_model_var.get(),
                no_benchmark=False,
                data_dir=Path("data"),
                device=self.device_var.get(),
                benchmark_split=args.benchmark_split,
                benchmark_count=int(self.benchmark_count_var.get()),
                benchmark_seed=int(self.benchmark_seed_var.get()),
                benchmark_indices=None,
                seq_len=int(self.seq_len_var.get()) if self.seq_len_var.get().strip() else None,
                stride=int(self.stride_var.get()) if self.stride_var.get().strip() else None,
                model_name=self.model_name_var.get().strip() or None,
                device_dequantize=args.device_dequantize,
            )
            return namespace

        def merge_selected(self) -> None:
            if self.report is None or self.bundle_a is None or self.bundle_b is None:
                messagebox.showwarning("No models", "Load two checkpoints first")
                return
            selected_paths = [self.tree_path_map[item] for item in self.tree.selection()]
            if not selected_paths:
                selected_paths = [""]
            merge_args = self._build_merge_args(selected_paths)
            try:
                run_dir, _merged_path, sequence_meta, benchmark_results = execute_merge_run(
                    self.bundle_a,
                    self.bundle_b,
                    self.report,
                    merge_args,
                )
            except Exception as exc:
                messagebox.showerror("Merge failed", str(exc))
                self.status_var.set(f"Merge failed: {exc}")
                return
            self.last_run_dir = run_dir
            self.last_benchmark_results = benchmark_results
            self.last_sequence_meta = sequence_meta
            self.status_var.set(f"Merge complete: {run_dir}")
            self.update_results()
            messagebox.showinfo("Merge complete", f"Artifacts written to\n{run_dir}")

        def update_results(self) -> None:
            if not self.last_run_dir or not self.last_benchmark_results:
                self._set_text(self.results_text, "No merge run completed yet.")
                return
            lines = [f"Run dir: {self.last_run_dir}"]
            for label, result in self.last_benchmark_results.items():
                lines.append(
                    f"{label}: pos_mse={result.summary.get('pos_mse_mean', float('nan')):.6f} "
                    f"vel_mae={result.summary.get('vel_mae_mean', float('nan')):.6f} "
                    f"seconds/seq={result.summary.get('seconds_per_sequence', float('nan')):.4f}"
                )
            self._set_text(self.results_text, "\n".join(lines))
            self._set_canvas_figure(
                self.results_canvas,
                make_benchmark_summary_figure(self.last_benchmark_results),
            )

    root = tk.Tk()
    app = MergeGuiApp(root)
    root.mainloop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    if args.command == "compare":
        return command_compare(args)
    if args.command == "merge":
        return command_merge(args)
    if args.command == "gui":
        return launch_gui(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())