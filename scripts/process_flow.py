"""Background script: extract optical flow for approved scenes that have keypoints.

Watches data/processed/ for approved scenes with keypoints that don't yet have
flow data for the configured method/features/scale combination.

Usage:
    python scripts/process_flow.py                   # process once and exit
    python scripts/process_flow.py --watch           # poll every 30 s
    python scripts/process_flow.py --device cuda:1   # GPU for RAFT
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import Config
from src.data.curation import (
    discover_scenes, flow_path, dense_flow_path,
    resolve_keypoints_path, resolve_flow_path, resolve_dense_flow_path,
)
from src.data.flow import compute_flow_for_video
from src.data.video import VideoReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


def process_scene(
    scene_dir: Path,
    preprocessed_dir: Path,
    cfg: Config,
    device: str,
    overwrite: bool = False,
) -> bool:
    """Extract optical flow for one scene. Returns True on success."""
    scene_id = scene_dir.name
    video_path = preprocessed_dir / f"{scene_id}.mp4"
    if not video_path.exists():
        log.warning("SKIP %s — no preprocessed video", scene_id)
        return False

    out_path = flow_path(scene_dir, cfg.flow.method,
                         cfg.flow.output_features, cfg.flow.scale)
    existing = resolve_flow_path(scene_dir, cfg.flow.method,
                                 cfg.flow.output_features, cfg.flow.scale)

    # Check dense flow existence
    need_dense = cfg.flow.save_dense and cfg.flow.method == "raft"
    dense_out = None
    if need_dense:
        dense_out = dense_flow_path(scene_dir, cfg.flow.method,
                                    cfg.flow.dense_size, cfg.flow.scale)
        existing_dense = resolve_dense_flow_path(scene_dir, cfg.flow.method,
                                                 cfg.flow.dense_size, cfg.flow.scale)
    else:
        existing_dense = None

    summary_done = existing is not None and not overwrite
    dense_done = (not need_dense) or (existing_dense is not None and not overwrite)

    if summary_done and dense_done:
        log.info("SKIP %s — flow already exists", scene_id)
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    flow_size = max(64, int(cfg.flow.scale * cfg.video.frame_size))

    log.info("Extracting flow: %s  method=%s  device=%s  size=%d  dense=%s",
             scene_id, cfg.flow.method, device, flow_size, need_dense)

    with VideoReader(video_path, vr_mode=False, target_size=flow_size) as reader:
        result = compute_flow_for_video(
            reader,
            output_features=cfg.flow.output_features,
            batch_size=60,
            method=cfg.flow.method,
            device=device,
            dense=need_dense,
            dense_size=cfg.flow.dense_size if need_dense else 32,
        )

    if need_dense:
        summary_data, dense_data = result
    else:
        summary_data = result

    if not summary_done:
        np.save(out_path, summary_data)
        log.info("Saved summary flow: %s  shape=%s", out_path, summary_data.shape)

    if need_dense and not dense_done:
        # Save as float16 to reduce disk usage
        dense_save = dense_data.astype(np.float16)
        np.save(dense_out, dense_save)
        log.info("Saved dense flow: %s  shape=%s  dtype=%s",
                 dense_out, dense_save.shape, dense_save.dtype)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Background optical flow extractor")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None,
                        help="Override computation device (e.g. cuda:1 for RAFT)")
    parser.add_argument("--watch", action="store_true",
                        help="Keep running, re-scan on --poll-interval")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if flow file already exists")
    parser.add_argument("--scenes", nargs="*",
                        help="Only process these scene IDs (default: all pending)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    data_dir = Path(args.data_dir)
    preprocessed_dir = data_dir / "preprocessed"
    processed_dir = data_dir / "processed"

    # Default device: CUDA for RAFT, CPU for farneback
    if args.device:
        device = args.device
    elif cfg.flow.method == "raft":
        device = "cuda:0"
    else:
        device = "cpu"

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
            # Must have keypoints before we extract flow
            if resolve_keypoints_path(sd, cfg.pose.model_name) is None:
                continue
            if not args.overwrite:
                summary_exists = resolve_flow_path(sd, cfg.flow.method,
                                                   cfg.flow.output_features, cfg.flow.scale) is not None
                need_dense = cfg.flow.save_dense and cfg.flow.method == "raft"
                dense_exists = (not need_dense) or (resolve_dense_flow_path(
                    sd, cfg.flow.method, cfg.flow.dense_size, cfg.flow.scale) is not None)
                if summary_exists and dense_exists:
                    continue
            pending.append(sid)

        log.info("Found %d scenes needing flow", len(pending))
        ok = failed = 0
        for sid in pending:
            try:
                if process_scene(processed_dir / sid, preprocessed_dir,
                                 cfg, device, args.overwrite):
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
