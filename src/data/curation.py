"""Curation state management and standardised path resolution for processed scenes.

Every processed scene can optionally have a review.json that tracks which stage
of the dataset pipeline it has reached and whether it is approved for training.

Storage layout (new):
    processed/<scene>/keypoints/<model_stem>.npy   – pose keypoints per model
    processed/<scene>/embeddings/<model_stem>.npy  – YOLO image embeddings
    processed/<scene>/flow/<method>_f<n>_s<scale>.npy – optical flow features
    processed/<scene>/review.json                  – curation state

Legacy paths (backward compat, checked if new paths absent):
    processed/<scene>/pose_keypoints.npy
    processed/<scene>/embeddings.npy
    processed/<scene>/optical_flow.npy
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .embeddings import validate_embeddings_file

# ── Path helpers ──────────────────────────────────────────────────────────────

def _model_stem(model_name: str) -> str:
    """Strip directory and extension from a model name to get a safe filename stem."""
    return Path(model_name).stem


def _flow_stem(method: str, output_features: int, scale: float) -> str:
    """Canonical filename stem for a flow config, e.g. 'raft_f64_s0.5'."""
    scale_s = f"{scale:.3f}".rstrip("0").rstrip(".")
    return f"{method}_f{output_features}_s{scale_s}"


def keypoints_path(scene_dir: Path, model_name: str) -> Path:
    """New path: keypoints/<model_stem>.npy"""
    return scene_dir / "keypoints" / f"{_model_stem(model_name)}.npy"


def embeddings_path(scene_dir: Path, model_name: str) -> Path:
    """New path: embeddings/<model_stem>.npy"""
    return scene_dir / "embeddings" / f"{_model_stem(model_name)}.npy"


def flow_path(scene_dir: Path, method: str, output_features: int, scale: float) -> Path:
    """New path: flow/<method>_f<n>_s<scale>.npy"""
    return scene_dir / "flow" / f"{_flow_stem(method, output_features, scale)}.npy"


# ── Backward-compat resolved paths ────────────────────────────────────────────

def resolve_keypoints_path(scene_dir: Path, model_name: str) -> Path | None:
    """Return existing keypoints path: new layout first, then legacy root file."""
    p = keypoints_path(scene_dir, model_name)
    if p.exists():
        return p
    legacy = scene_dir / "pose_keypoints.npy"
    return legacy if legacy.exists() else None


def resolve_embeddings_path(scene_dir: Path, model_name: str) -> Path | None:
    """Return existing embeddings path: new layout first, then legacy root file."""
    return inspect_embeddings_path(scene_dir, model_name)[0]


def inspect_embeddings_path(
    scene_dir: Path,
    model_name: str,
    max_persons: int | None = None,
    require_current: bool = False,
) -> tuple[Path | None, str]:
    """Resolve embeddings path and optionally require the current cache format."""
    candidates = [embeddings_path(scene_dir, model_name), scene_dir / "embeddings.npy"]
    reasons: list[str] = []

    for candidate in candidates:
        if not candidate.exists():
            continue
        if not require_current:
            return candidate, "ok"

        valid, reason = validate_embeddings_file(candidate, max_persons=max_persons)
        if valid:
            return candidate, "ok"
        reasons.append(f"{candidate.name}: {reason}")

    if reasons:
        return None, "; ".join(reasons)
    return None, "missing"


def resolve_flow_path(scene_dir: Path, method: str, output_features: int, scale: float) -> Path | None:
    """Return existing flow path: new layout first, then legacy root file."""
    p = flow_path(scene_dir, method, output_features, scale)
    if p.exists():
        return p
    legacy = scene_dir / "optical_flow.npy"
    return legacy if legacy.exists() else None


# ── Review state ──────────────────────────────────────────────────────────────

REVIEW_FILENAME = "review.json"

CurationStatus = Literal["pending", "approved", "rejected"]

_DEFAULT_REVIEW: dict = {
    "status": "pending",        # stage-1: initial content review
    "stage2_status": "pending", # stage-2: review after keypoint inspection
    "force_val": False,         # force this scene into the validation split
    "imported_at": None,
    "reviewed_at": None,
    "stage2_reviewed_at": None,
    "notes": "",
}


def read_review(scene_dir: Path) -> dict:
    """Load curation state for a scene. Returns defaults if review.json is absent."""
    path = scene_dir / REVIEW_FILENAME
    if not path.exists():
        return dict(_DEFAULT_REVIEW)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        for k, v in _DEFAULT_REVIEW.items():
            state.setdefault(k, v)
        return state
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_REVIEW)


def write_review(scene_dir: Path, state: dict) -> None:
    """Persist curation state to review.json."""
    (scene_dir / REVIEW_FILENAME).write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def approve(scene_dir: Path) -> None:
    state = read_review(scene_dir)
    state["status"] = "approved"
    state["reviewed_at"] = _now_iso()
    write_review(scene_dir, state)


def reject(scene_dir: Path) -> None:
    state = read_review(scene_dir)
    state["status"] = "rejected"
    state["reviewed_at"] = _now_iso()
    write_review(scene_dir, state)


def approve_stage2(scene_dir: Path) -> None:
    state = read_review(scene_dir)
    state["stage2_status"] = "approved"
    state["stage2_reviewed_at"] = _now_iso()
    write_review(scene_dir, state)


def reject_stage2(scene_dir: Path) -> None:
    state = read_review(scene_dir)
    state["stage2_status"] = "rejected"
    state["stage2_reviewed_at"] = _now_iso()
    write_review(scene_dir, state)


def set_force_val(scene_dir: Path, val: bool) -> None:
    state = read_review(scene_dir)
    state["force_val"] = val
    write_review(scene_dir, state)


# ── Scene stage summary ───────────────────────────────────────────────────────

def get_scene_stage(
    scene_dir: Path,
    model_name: str,
    flow_method: str,
    flow_output_features: int,
    flow_scale: float,
) -> str:
    """Return a short string describing the highest processing stage reached.

    Values (in ascending order of completeness):
        "rejected"   – rejected at any stage
        "pending"    – needs stage-1 review
        "approved"   – stage-1 approved, keypoints not yet extracted
        "keypoints"  – keypoints extracted, pending stage-2 review
        "stage2_ok"  – stage-2 approved; flow may or may not be present
        "flow"       – stage-2 approved + flow extracted
        "legacy"     – no review.json; pre-curation data (treated as approved)
    """
    review_path = scene_dir / REVIEW_FILENAME
    if not review_path.exists():
        return "legacy"

    state = read_review(scene_dir)
    if state["status"] == "rejected" or state["stage2_status"] == "rejected":
        return "rejected"
    if state["status"] == "pending":
        return "pending"

    has_kpts = resolve_keypoints_path(scene_dir, model_name) is not None
    has_flow = resolve_flow_path(scene_dir, flow_method, flow_output_features, flow_scale) is not None

    if state["stage2_status"] == "approved" and has_flow:
        return "flow"
    if state["stage2_status"] == "approved":
        return "stage2_ok"
    if has_kpts:
        return "keypoints"
    if state["status"] == "approved":
        return "approved"
    return "pending"


# ── Scene discovery ───────────────────────────────────────────────────────────

def discover_scenes(
    data_dir: Path,
    include_rejected: bool = False,
    require_labels: bool = True,
) -> list[tuple[str, dict]]:
    """Discover all scenes, returning (scene_id, review_state) pairs sorted by name."""
    processed = data_dir / "processed"
    if not processed.exists():
        return []
    results = []
    for d in sorted(processed.iterdir()):
        if not d.is_dir():
            continue
        if require_labels and not (d / "labels.npy").exists():
            continue
        state = read_review(d)
        if not include_rejected and state["status"] == "rejected":
            continue
        results.append((d.name, state))
    return results
