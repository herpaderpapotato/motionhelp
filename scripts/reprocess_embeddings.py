"""Re-extract embeddings for all processed scenes using SinglePassExtractor.

Replaces model.embed() embeddings with hook-based RoI Align per-person
embeddings. Also re-extracts keypoints for consistency (same predict() call
produces both). Existing flow and labels are preserved.

Usage:
    python scripts/reprocess_embeddings.py --data-dir data
    python scripts/reprocess_embeddings.py --data-dir data --dry-run
    python scripts/reprocess_embeddings.py --data-dir data --scenes scene_00018_t00799_40s scene_00018_t00926_40s
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import random

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.pose import load_pose_model
from src.data.extraction import SinglePassExtractor, extract_single_pass_batched, BACKBONE_STRIDE

log = logging.getLogger(__name__)

KP_FILE = "keypoints/pose-vrlens-finetunes-large.npy"
EMB_FILE = "embeddings/pose-vrlens-finetunes-large.npy"


def reprocess_scene(
    scene_id: str,
    data_dir: Path,
    extractor: SinglePassExtractor,
    batch_size: int = 32,
    dry_run: bool = False,
) -> bool:
    """Re-extract keypoints and embeddings for a single scene.

    Loads the preprocessed video, runs single-pass extraction, and overwrites
    the keypoints and embeddings .npy files.

    Returns True on success, False on skip/error.
    """
    video_path = data_dir / "preprocessed" / f"{scene_id}.mp4"
    scene_dir = data_dir / "processed" / scene_id

    if not video_path.exists():
        log.warning("No video for %s, skipping", scene_id)
        return False

    labels_path = scene_dir / "labels.npy"
    if not labels_path.exists():
        log.warning("No labels for %s, skipping", scene_id)
        return False

    n_frames = np.load(str(labels_path), mmap_mode="r").shape[0]

    if dry_run:
        log.info("[DRY RUN] Would reprocess %s (%d frames)", scene_id, n_frames)
        return True

    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(video_path), device="cuda", dimension_order="NHWC")
        frames = decoder.get_frames_in_range(0, min(n_frames, len(decoder)))
        frames_np = frames.data.cpu().numpy()  # [N, 640, 640, 3]
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames_list = []
        while len(frames_list) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames_np = np.stack(frames_list)

    actual_n = min(len(frames_np), n_frames)
    frames_np = frames_np[:actual_n]

    keypoints, embeddings = extract_single_pass_batched(
        extractor, list(frames_np), batch_size,
    )

    # Save
    kp_dir = scene_dir / "keypoints"
    emb_dir = scene_dir / "embeddings"
    kp_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(scene_dir / KP_FILE), keypoints)
    np.save(str(scene_dir / EMB_FILE), embeddings)

    # Save embedding metadata for validation
    meta = {
        "format_version": 3,
        "method": "single_pass_hook_roi_align",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "embed_dim": 512,
        "max_persons": extractor.max_persons,
    }
    meta_path = (scene_dir / EMB_FILE).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract embeddings using single-pass hook approach",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pose-model", type=Path,
                        default=Path("data/models/pose/pose-vrlens-finetunes-large.pt"))
    parser.add_argument("--max-persons", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="List scenes to process without modifying files")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Specific scene IDs to process (default: all)")
    parser.add_argument("--split", default=None,
                        help="Process only scenes in this split (train/val/test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Determine which scenes to process
    if args.scenes:
        scene_ids = args.scenes
    elif args.split:
        split_path = args.data_dir / "splits" / f"{args.split}.json"
        with open(split_path) as f:
            scene_ids = json.load(f)
        log.info("Processing %d scenes from %s split", len(scene_ids), args.split)
    else:
        processed_dir = args.data_dir / "processed"
        scene_ids = sorted(
            d.name for d in processed_dir.iterdir()
            if d.is_dir() and (d / "labels.npy").exists()
        )
        log.info("Processing all %d scenes", len(scene_ids))

    # Load model
    device = torch.device(args.device)
    log.info("Loading pose model from %s...", args.pose_model)
    pose_model = load_pose_model(
        model_name="yolo11m-pose",
        model_path=str(args.pose_model),
        device=str(device),
    )
    extractor = SinglePassExtractor(
        pose_model, max_persons=args.max_persons, n_keypoints=21,
        confidence_threshold=0.02, device=str(device),
    )
    log.info(
        "Extractor: layer %d (%s), stride %d",
        extractor.layer_idx,
        type(pose_model.model.model[extractor.layer_idx]).__name__,
        BACKBONE_STRIDE if hasattr(extractor, '_hook') else 0,
    )

    # Process scenes
    success = 0
    failed = 0
    t_start = time.perf_counter()

    random.shuffle(scene_ids)

    for i, scene_id in enumerate(scene_ids):
        # check if embeddings\pose-vrlens-finetunes-large.json says "method": "ultralytics_model.embed" or if "method": "single_pass_hook_roi_align" and skip if already "single_pass_hook_roi_align"
        meta_path = args.data_dir / "processed" / scene_id / "embeddings" / "pose-vrlens-finetunes-large.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("method") == "single_pass_hook_roi_align":
                log.info("[%d/%d] %s already processed with single-pass method, skipping",
                         i + 1, len(scene_ids), scene_id)
                success += 1
                continue

        t0 = time.perf_counter()
        try:
            ok = reprocess_scene(
                scene_id, args.data_dir, extractor,
                batch_size=args.batch_size, dry_run=args.dry_run,
            )
            dt = time.perf_counter() - t0
            if ok:
                success += 1
                if not args.dry_run:
                    log.info(
                        "[%d/%d] %s done in %.1fs",
                        i + 1, len(scene_ids), scene_id, dt,
                    )
            else:
                failed += 1
        except Exception as e:
            failed += 1
            log.error("[%d/%d] %s failed: %s", i + 1, len(scene_ids), scene_id, e)

    total_time = time.perf_counter() - t_start
    extractor.close()

    log.info(
        "Done: %d success, %d failed, %.1fs total (%.2fs/scene avg)",
        success, failed, total_time,
        total_time / max(success, 1),
    )


if __name__ == "__main__":
    main()
