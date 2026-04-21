"""Build train/val splits for TCN and disposition models.

Reads processed scene directories, applies quality filtering, and writes the
requested split files under data/splits/.

Usage:
    python scripts/build_splits.py --val-scenes scene_00045
    python scripts/build_splits.py --target disposition --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.curation import inspect_embeddings_path, inspect_model_embeddings_path, read_review

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DISPOSITION_MODEL = "vrlens-finetunes-multiclass-v2-yolo26m-pose"


@dataclass(frozen=True)
class SplitTarget:
    name: str
    train_filename: str
    val_filename: str
    model_name: str
    strict_model_embeddings: bool


def resolve_split_target(
    target: str,
    cfg: Config,
    model_name: str | None = None,
) -> SplitTarget:
    if target == "disposition":
        return SplitTarget(
            name=target,
            train_filename="disposition_train.json",
            val_filename="disposition_val.json",
            model_name=model_name or DEFAULT_DISPOSITION_MODEL,
            strict_model_embeddings=True,
        )
    if target == "default":
        return SplitTarget(
            name=target,
            train_filename="train.json",
            val_filename="val.json",
            model_name=model_name or cfg.pose.model_name,
            strict_model_embeddings=False,
        )
    raise ValueError(f"Unsupported split target: {target}")


def inspect_required_embeddings(
    scene_dir: Path,
    *,
    model_name: str,
    max_persons: int,
    strict_model_embeddings: bool,
) -> tuple[Path | None, str]:
    if strict_model_embeddings:
        return inspect_model_embeddings_path(
            scene_dir,
            model_name,
            max_persons=max_persons,
        )

    return inspect_embeddings_path(
        scene_dir,
        model_name,
        max_persons=max_persons,
        require_current=True,
    )


def scene_is_rejected(scene_dir: Path) -> bool:
    review = read_review(scene_dir)
    return (
        review.get("status") == "rejected"
        or review.get("stage2_status") == "rejected"
    )


def collect_split_candidates(
    processed_dir: Path,
    *,
    model_name: str,
    max_persons: int,
    require_embeddings: bool,
    strict_model_embeddings: bool,
) -> tuple[list[str], dict[str, dict[str, float]], list[tuple[str, str]]]:
    candidate_ids: list[str] = []
    stats: dict[str, dict[str, float]] = {}
    skipped: list[tuple[str, str]] = []

    for scene_dir in sorted(processed_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        labels_path = scene_dir / "labels.npy"
        if not labels_path.exists():
            skipped.append((scene_dir.name, "missing labels.npy"))
            continue

        if scene_is_rejected(scene_dir):
            skipped.append((scene_dir.name, "scene is rejected in review.json"))
            continue

        if require_embeddings:
            emb_path, emb_reason = inspect_required_embeddings(
                scene_dir,
                model_name=model_name,
                max_persons=max_persons,
                strict_model_embeddings=strict_model_embeddings,
            )
            if emb_path is None:
                skipped.append((scene_dir.name, emb_reason))
                continue

        labels = np.load(labels_path)
        stats[scene_dir.name] = {
            "frames": int(len(labels)),
            "label_std": float(np.std(labels)),
            "label_range": float(np.ptp(labels)),
            "label_mean": float(np.mean(labels)),
        }
        candidate_ids.append(scene_dir.name)

    return candidate_ids, stats, skipped


def filter_split_candidates(
    candidate_ids: list[str],
    stats: dict[str, dict[str, float]],
    *,
    min_label_std: float,
    min_frames: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    filtered_ids: list[str] = []
    rejected: list[tuple[str, str]] = []

    for scene_id in candidate_ids:
        scene_stats = stats[scene_id]
        if scene_stats["label_std"] < min_label_std:
            rejected.append(
                (scene_id, f"label_std={scene_stats['label_std']:.4f} < {min_label_std}")
            )
            continue
        if scene_stats["frames"] < min_frames:
            rejected.append((scene_id, f"frames={scene_stats['frames']} < {min_frames}"))
            continue
        filtered_ids.append(scene_id)

    return filtered_ids, rejected


def build_train_val_split(
    filtered_ids: list[str],
    *,
    val_scenes: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    val_scene_set = set(val_scenes)
    val_ids = [scene_id for scene_id in filtered_ids if scene_id in val_scene_set]
    missing_val_scenes = [scene_id for scene_id in val_scenes if scene_id not in filtered_ids]

    remaining_ids = [scene_id for scene_id in filtered_ids if scene_id not in val_scene_set]
    n_min_val = max(1, math.ceil(len(filtered_ids) * val_ratio)) if filtered_ids else 0
    if len(val_ids) < n_min_val:
        to_add = n_min_val - len(val_ids)
        rng = random.Random(seed)
        rng.shuffle(remaining_ids)
        extra_val = remaining_ids[:to_add]
        val_ids.extend(extra_val)
        remaining_ids = remaining_ids[to_add:]
        log.info(
            "Allocated %d additional scenes to validation to meet min ratio %.2f",
            len(extra_val),
            val_ratio,
        )

    return sorted(remaining_ids), sorted(val_ids), missing_val_scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val splits with quality filtering")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--target",
        choices=["default", "disposition"],
        default="default",
        help="Which split files to build",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Embedding model name to require for included scenes",
    )
    parser.add_argument(
        "--val-scenes",
        nargs="+",
        default=["scene_00045"],
        help="Scene IDs to reserve for validation",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Minimum fraction of filtered scenes to allocate to validation",
    )
    parser.add_argument(
        "--min-label-std",
        type=float,
        default=0.02,
        help="Minimum label std dev to include a scene (filters dead segments)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=200,
        help="Minimum number of frames to include a scene",
    )
    parser.add_argument(
        "--require-embeddings",
        action="store_true",
        default=True,
        help="Only include scenes with valid embeddings for the selected model (default: True)",
    )
    parser.add_argument(
        "--no-require-embeddings",
        dest="require_embeddings",
        action="store_false",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    target = resolve_split_target(args.target, cfg, args.model_name)

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"

    if not processed_dir.exists():
        log.error("No processed directory found at %s", processed_dir)
        return

    log.info("Building %s splits", target.name)
    log.info("  Data directory: %s", data_dir)
    log.info("  Output files: %s, %s", target.train_filename, target.val_filename)
    log.info("  Embedding model: %s", target.model_name)
    log.info("  Validation scenes: %s", args.val_scenes)
    log.info("  Minimum validation ratio: %.2f", args.val_ratio)
    log.info("  Minimum label std dev: %.4f", args.min_label_std)
    log.info("  Minimum frames: %d", args.min_frames)
    log.info("  Require embeddings: %s", args.require_embeddings)
    log.info("  Strict model embeddings: %s", target.strict_model_embeddings)
    log.info("  Random seed: %d", args.seed)
    log.info("  Dry run: %s", args.dry_run)

    candidate_ids, stats, skipped = collect_split_candidates(
        processed_dir,
        model_name=target.model_name,
        max_persons=cfg.model.max_persons,
        require_embeddings=args.require_embeddings,
        strict_model_embeddings=target.strict_model_embeddings,
    )
    log.info("Found %d candidate scenes", len(candidate_ids))

    if skipped:
        log.info("Skipped %d scenes before quality filtering", len(skipped))
        for scene_id, reason in skipped:
            log.info("  %s: %s", scene_id, reason)

    filtered_ids, rejected = filter_split_candidates(
        candidate_ids,
        stats,
        min_label_std=args.min_label_std,
        min_frames=args.min_frames,
    )

    if rejected:
        log.info("Rejected %d scenes during quality filtering", len(rejected))
        for scene_id, reason in rejected:
            log.info("  %s: %s", scene_id, reason)

    log.info("After filtering: %d scenes", len(filtered_ids))

    train_ids, val_ids, missing_val_scenes = build_train_val_split(
        filtered_ids,
        val_scenes=args.val_scenes,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if missing_val_scenes:
        log.warning(
            "Requested validation scenes were filtered out or not found: %s",
            missing_val_scenes,
        )

    train_frames = sum(stats[scene_id]["frames"] for scene_id in train_ids)
    val_frames = sum(stats[scene_id]["frames"] for scene_id in val_ids)
    log.info("Train: %d scenes (%d frames)", len(train_ids), train_frames)
    log.info("Val: %d scenes (%d frames)", len(val_ids), val_frames)

    log.info("--- Train scenes ---")
    for scene_id in train_ids:
        scene_stats = stats[scene_id]
        log.info(
            "  %s: %d frames, std=%.3f, range=%.2f",
            scene_id,
            scene_stats["frames"],
            scene_stats["label_std"],
            scene_stats["label_range"],
        )
    log.info("--- Val scenes ---")
    for scene_id in val_ids:
        scene_stats = stats[scene_id]
        log.info(
            "  %s: %d frames, std=%.3f, range=%.2f",
            scene_id,
            scene_stats["frames"],
            scene_stats["label_std"],
            scene_stats["label_range"],
        )

    if args.dry_run:
        log.info("Dry run - not writing splits")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / target.train_filename, "w", encoding="utf-8") as handle:
        json.dump(train_ids, handle, indent=2)
    with open(splits_dir / target.val_filename, "w", encoding="utf-8") as handle:
        json.dump(val_ids, handle, indent=2)

    log.info(
        "Wrote %s (%d) and %s (%d)",
        target.train_filename,
        len(train_ids),
        target.val_filename,
        len(val_ids),
    )


if __name__ == "__main__":
    main()
