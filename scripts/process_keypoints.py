"""Background script: extract keypoints + embeddings for approved scenes.

Watches data/processed/ for scenes with review status='approved' that don't
yet have keypoint data for the configured model, then extracts them.

Usage:
    python scripts/process_keypoints.py                  # process once and exit
    python scripts/process_keypoints.py --watch          # poll every 30 s
    python scripts/process_keypoints.py --device cuda:1  # use specific GPU
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import Config
from src.data.curation import (
    discover_scenes, keypoints_path, embeddings_path,
    inspect_embeddings_path, resolve_keypoints_path, resolve_embeddings_path,
)
from src.data.pose import load_pose_model, extract_pose_video
from src.data.embeddings import extract_embeddings_video, save_embeddings_artifacts
from src.data.extraction import SinglePassExtractor, extract_single_pass_batched

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

# Multiclass file names (must match reprocess_embeddings.py)
KP_FILE_MULTICLASS = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
EMB_FILE_MULTICLASS = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
MULTICLASS_POSE_MODEL = "data/models/pose/vrlens-finetunes-multiclass-v2-yolo11m-pose.pt"


def process_scene(
    scene_dir: Path,
    preprocessed_dir: Path,
    cfg: Config,
    pose_model,
    overwrite: bool = False,
) -> bool:
    """Extract keypoints and embeddings for one scene. Returns True on success."""
    scene_id = scene_dir.name
    video_path = preprocessed_dir / f"{scene_id}.mp4"
    if not video_path.exists():
        log.warning("SKIP %s — no preprocessed video at %s", scene_id, video_path)
        return False

    kpts_out = keypoints_path(scene_dir, cfg.pose.model_name)
    emb_out = embeddings_path(scene_dir, cfg.pose.model_name)

    # Verify expected frame count from metadata if present
    expected_frames: int | None = None
    meta_path = scene_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        expected_frames = meta.get("total_frames")

    # ── Extract keypoints ──────────────────────────────────────────────────
    kpts_existing = resolve_keypoints_path(scene_dir, cfg.pose.model_name)
    if kpts_existing is None or overwrite:
        log.info("Extracting keypoints: %s", scene_id)
        kpts_out.parent.mkdir(parents=True, exist_ok=True)
        pose_data = extract_pose_video(
            pose_model,
            video_path,
            vr_mode=cfg.video.vr_mode,
            sbs_crop=cfg.video.sbs_crop,
            frame_size=cfg.video.frame_size,
            batch_size=cfg.pose.batch_size,
            max_persons=cfg.model.max_persons,
            confidence_threshold=cfg.pose.confidence_threshold,
            n_keypoints=cfg.pose.n_keypoints,
        )
        np.save(kpts_out, pose_data)
        log.info("Saved keypoints: %s  shape=%s", kpts_out, pose_data.shape)
        if expected_frames and len(pose_data) != expected_frames:
            log.warning("Frame count mismatch: expected %d, got %d",
                        expected_frames, len(pose_data))
    else:
        log.info("SKIP keypoints %s — already exists at %s", scene_id, kpts_existing)

    # ── Extract embeddings ─────────────────────────────────────────────────
    emb_existing, emb_status = inspect_embeddings_path(
        scene_dir,
        cfg.pose.model_name,
        max_persons=cfg.model.max_persons,
        require_current=True,
    )
    if emb_existing is None or overwrite:
        if emb_status != "missing":
            log.info("Regenerating embeddings for %s (%s)", scene_id, emb_status)
        log.info("Extracting embeddings: %s", scene_id)
        emb_out.parent.mkdir(parents=True, exist_ok=True)
        embs = extract_embeddings_video(
            pose_model,
            video_path,
            batch_size=cfg.pose.batch_size,
            max_persons=cfg.model.max_persons,
            confidence_threshold=cfg.pose.confidence_threshold,
        )
        save_embeddings_artifacts(
            emb_out,
            embs,
            max_persons=cfg.model.max_persons,
            video_path=video_path,
        )
        det_rate = (embs[:, 0].sum(axis=-1) != 0).mean() * 100
        log.info("Saved embeddings: %s  shape=%s  det=%.1f%%",
                 emb_out, embs.shape, det_rate)
        if expected_frames and len(embs) != expected_frames:
            log.warning("Frame count mismatch: expected %d, got %d",
                        expected_frames, len(embs))
    else:
        log.info("SKIP embeddings %s — already exists at %s", scene_id, emb_existing)

    return True


def process_scene_multiclass(
    scene_dir: Path,
    preprocessed_dir: Path,
    extractor: SinglePassExtractor,
    batch_size: int = 32,
    overwrite: bool = False,
) -> bool:
    """Extract multiclass keypoints and embeddings using SinglePassExtractor."""
    scene_id = scene_dir.name
    video_path = preprocessed_dir / f"{scene_id}.mp4"
    if not video_path.exists():
        log.warning("SKIP %s — no preprocessed video at %s", scene_id, video_path)
        return False

    kp_out = scene_dir / KP_FILE_MULTICLASS
    emb_out = scene_dir / EMB_FILE_MULTICLASS

    # Check if already processed
    meta_path = emb_out.with_suffix(".json")
    if not overwrite and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("method") == "single_pass_hook_roi_align" and meta.get("multiclass", False):
            log.info("SKIP %s — already processed with multiclass method", scene_id)
            return True

    # Get expected frame count
    labels_path = scene_dir / "labels.npy"
    if not labels_path.exists():
        log.warning("SKIP %s — no labels.npy", scene_id)
        return False
    n_frames = np.load(str(labels_path), mmap_mode="r").shape[0]

    log.info("Extracting multiclass keypoints+embeddings: %s (%d frames)", scene_id, n_frames)

    # Load video frames
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(video_path), device="cuda", dimension_order="NHWC")
        frames = decoder.get_frames_in_range(0, min(n_frames, len(decoder)))
        frames_np = frames.data.cpu().numpy()
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
    kp_out.parent.mkdir(parents=True, exist_ok=True)
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(kp_out), keypoints)
    np.save(str(emb_out), embeddings)

    # Save metadata
    meta = {
        "format_version": 4,
        "method": "single_pass_hook_roi_align",
        "multiclass": True,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "embed_dim": 512,
        "max_persons": extractor.max_persons,
        "max_partners": extractor.max_partners,
        "max_beholders": extractor.max_beholders,
        "n_beholder_keypoints": extractor.n_beholder_keypoints,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Saved multiclass: kp=%s emb=%s", kp_out, emb_out)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Background keypoint + embedding extractor")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda:0)")
    parser.add_argument("--watch", action="store_true",
                        help="Keep running, re-scan on --poll-interval")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between scans in watch mode (default: 30)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if files already exist")
    parser.add_argument("--scenes", nargs="*",
                        help="Only process these scene IDs (default: all pending)")
    parser.add_argument("--multiclass", action="store_true",
                        help="Use multiclass model (partner + beholder)")
    parser.add_argument("--max-partners", type=int, default=5)
    parser.add_argument("--max-beholders", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.device:
        cfg.pose.device = args.device

    data_dir = Path(args.data_dir)
    preprocessed_dir = data_dir / "preprocessed"
    processed_dir = data_dir / "processed"

    # Load model
    if args.multiclass:
        import torch
        pose_model_path = MULTICLASS_POSE_MODEL
        log.info("Loading multiclass pose model: %s", pose_model_path)
        pose_model = load_pose_model(
            model_name="yolo11m-pose",
            model_path=pose_model_path,
            device=cfg.pose.device,
        )
        extractor = SinglePassExtractor(
            pose_model,
            max_persons=args.max_partners + args.max_beholders,
            n_keypoints=21,
            confidence_threshold=0.02,
            device=cfg.pose.device,
            multiclass=True,
            max_partners=args.max_partners,
            max_beholders=args.max_beholders,
        )
    else:
        log.info("Loading pose model: %s", cfg.pose.model_path or cfg.pose.model_name)
        pose_model = load_pose_model(
            model_name=cfg.pose.model_name,
            model_path=cfg.pose.model_path,
            device=cfg.pose.device,
        )

    def run_once() -> None:
        all_scenes = discover_scenes(data_dir, include_rejected=False, require_labels=True)
        pending = []
        for sid, state in all_scenes:
            # if state["status"] != "approved":
            #     continue
            if state["status"] == "rejected":
                continue
            sd = processed_dir / sid
            if args.scenes and sid not in args.scenes:
                continue

            if args.multiclass:
                # Check multiclass outputs
                meta_path = sd / "embeddings" / "vrlens-finetunes-multiclass-v2-yolo11m-pose.json"
                if not args.overwrite and meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("method") == "single_pass_hook_roi_align" and meta.get("multiclass"):
                        continue
                pending.append(sid)
            else:
                needs_kpts = resolve_keypoints_path(sd, cfg.pose.model_name) is None
                needs_emb = inspect_embeddings_path(
                    sd,
                    cfg.pose.model_name,
                    max_persons=cfg.model.max_persons,
                    require_current=True,
                )[0] is None
                if args.overwrite or needs_kpts or needs_emb:
                    pending.append(sid)

        log.info("Found %d scenes needing keypoints/embeddings", len(pending))
        ok = failed = 0
        for sid in pending:
            try:
                if args.multiclass:
                    success = process_scene_multiclass(
                        processed_dir / sid, preprocessed_dir, extractor,
                        batch_size=args.batch_size, overwrite=args.overwrite,
                    )
                else:
                    success = process_scene(
                        processed_dir / sid, preprocessed_dir, cfg,
                        pose_model, args.overwrite,
                    )
                if success:
                    ok += 1
                else:
                    failed += 1
            except Exception:
                log.exception("Failed: %s", sid)
                failed += 1
        log.info("Batch complete: %d ok, %d failed", ok, failed)

    if args.watch:
        log.info("Watch mode (poll every %ds) — Ctrl+C to stop", args.poll_interval)
        while True:
            run_once()
            time.sleep(args.poll_interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
