"""Build train/val splits combining full-video and segment data.

Reads all processed directories, applies quality filtering (e.g., minimum
label variance to exclude dead segments), and writes train.json / val.json.

Usage:
    python scripts/build_splits.py --val-scenes scene_00045
    python scripts/build_splits.py --val-ratio 0.15 --min-label-std 0.05
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.curation import inspect_embeddings_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build train/val splits with quality filtering")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--val-scenes", nargs="+", default=["scene_00045"],
                        help="Scene IDs to reserve for validation")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Minimum fraction of filtered scenes to allocate to validation")
    parser.add_argument("--min-label-std", type=float, default=0.02,
                        help="Minimum label std dev to include a scene (filters dead segments)")
    parser.add_argument("--min-frames", type=int, default=200,
                        help="Minimum number of frames to include a scene")
    parser.add_argument("--require-embeddings", action="store_true", default=True,
                        help="Only include scenes with current embeddings for the configured model (default: True)")
    parser.add_argument("--no-require-embeddings", dest="require_embeddings", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"

    if not processed_dir.exists():
        log.error("No processed directory found at %s", processed_dir)
        return

    print("Building train/val splits with the following parameters:")
    print(f"  Data directory: {data_dir}")
    print(f"  Validation scenes: {args.val_scenes}")
    print(f"  Minimum validation ratio: {args.val_ratio:.2f}")
    print(f"  Minimum label std dev: {args.min_label_std:.4f}")
    print(f"  Minimum frames: {args.min_frames}")
    print(f"  Require embeddings: {args.require_embeddings}")
    print(f"  Random seed: {args.seed}")
    print(f"  Dry run: {args.dry_run}")
    print()
    print(f"Found {len(list(processed_dir.iterdir()))} items in processed directory")
    # Discover all processed scenes/segments
    all_ids = []
    stats = {}
    for d in sorted(processed_dir.iterdir()):
        if not d.is_dir():
            continue
        labels_path = d / "labels.npy"
        if not labels_path.exists():
            log.warning("No labels for %s, skipping", d.name)
            continue
        if args.require_embeddings and inspect_embeddings_path(
            d,
            cfg.pose.model_name,
            max_persons=cfg.model.max_persons,
            require_current=True,
        )[0] is None:
            log.debug("No current embeddings for %s, skipping", d.name)
            continue

        labels = np.load(labels_path)
        label_std = float(np.std(labels))
        label_range = float(np.ptp(labels))
        n_frames = len(labels)

        stats[d.name] = {
            "frames": n_frames,
            "label_std": label_std,
            "label_range": label_range,
            "label_mean": float(np.mean(labels)),
        }
        all_ids.append(d.name)

    log.info("Found %d processed scenes/segments", len(all_ids))

    # Apply quality filters
    filtered_ids = []
    rejected = []
    for vid_id in all_ids:
        s = stats[vid_id]
        if s["label_std"] < args.min_label_std:
            rejected.append((vid_id, f"label_std={s['label_std']:.4f} < {args.min_label_std}"))
            continue
        if s["frames"] < args.min_frames:
            rejected.append((vid_id, f"frames={s['frames']} < {args.min_frames}"))
            continue
        filtered_ids.append(vid_id)

    if rejected:
        log.info("Rejected %d scenes:", len(rejected))
        for vid_id, reason in rejected:
            log.info("  %s: %s", vid_id, reason)

    log.info("After filtering: %d scenes", len(filtered_ids))

    # Split into train/val
    val_scene_set = set(args.val_scenes)
    val_ids = [v for v in filtered_ids if v in val_scene_set]
    missing_val_scenes = [v for v in args.val_scenes if v not in filtered_ids]
    if missing_val_scenes:
        log.warning("Requested validation scenes were filtered out or not found: %s", missing_val_scenes)

    remaining_ids = [v for v in filtered_ids if v not in val_scene_set]
    n_min_val = max(1, math.ceil(len(filtered_ids) * args.val_ratio)) if filtered_ids else 0
    if len(val_ids) < n_min_val:
        to_add = n_min_val - len(val_ids)
        rng = random.Random(args.seed)
        rng.shuffle(remaining_ids)
        extra_val = remaining_ids[:to_add]
        val_ids.extend(extra_val)
        remaining_ids = remaining_ids[to_add:]
        log.info("Allocated %d additional scenes to validation to meet min ratio %.2f", len(extra_val), args.val_ratio)

    val_ids = sorted(val_ids)
    train_ids = sorted(remaining_ids)

    # Summary
    train_frames = sum(stats[v]["frames"] for v in train_ids)
    val_frames = sum(stats[v]["frames"] for v in val_ids)

    log.info("Train: %d scenes (%d frames)", len(train_ids), train_frames)
    log.info("Val: %d scenes (%d frames)", len(val_ids), val_frames)

    # Show all scenes with stats
    log.info("--- Train scenes ---")
    for vid_id in sorted(train_ids):
        s = stats[vid_id]
        log.info("  %s: %d frames, std=%.3f, range=%.2f", vid_id, s["frames"], s["label_std"], s["label_range"])
    log.info("--- Val scenes ---")
    for vid_id in sorted(val_ids):
        s = stats[vid_id]
        log.info("  %s: %d frames, std=%.3f, range=%.2f", vid_id, s["frames"], s["label_std"], s["label_range"])

    if args.dry_run:
        log.info("Dry run — not writing splits")
        return

    # Write splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "train.json", "w") as f:
        json.dump(sorted(train_ids), f, indent=2)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(sorted(val_ids), f, indent=2)

    log.info("Wrote train.json (%d) and val.json (%d)", len(train_ids), len(val_ids))


if __name__ == "__main__":
    main()
