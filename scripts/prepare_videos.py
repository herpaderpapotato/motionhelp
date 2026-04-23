"""Batch import: preprocess video segments from the xbvr library and extract labels.

This is the first step in the curation pipeline.  It does NOT extract pose
keypoints or optical flow — those are handled by the dedicated background
scripts process_keypoints.py and process_flow.py.

Each processed segment lands in data/processed/<scene_id>/ with:
    metadata.json   – timing, resolution, fps
    labels.npy      – per-frame funscript position in [0, 1]
    review.json     – status = "pending"  (ready for stage-1 review)

The preprocessed video is saved to data/preprocessed/<scene_id>.mp4.

Usage:
    # Pull 3 random 40-second segments each from up to 50 new videos:
    python scripts/prepare_videos.py

    # Different segment length / count:
    python scripts/prepare_videos.py --duration 60 --segments-per-video 2

    # Keep exact source timing but drop every other frame:
    python scripts/prepare_videos.py --half-rate

    # Multi-machine coordination (lock files prevent duplicate work):
    python scripts/prepare_videos.py --lock-dir data/locks

    # Force re-import scenes that already exist:
    python scripts/prepare_videos.py --overwrite
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.curation import read_review, write_review
from src.data.datasource import pairs_from_xbvr
from src.data.funscript import load_funscript, get_actions, actions_to_timestamps
from src.data.preprocess import probe_video, build_preprocess_command
from src.data.video import get_video_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_database_url() -> str:
    from dotenv import dotenv_values
    env = dotenv_values(Path(__file__).parent.parent / ".env")
    url = env.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not found in .env file")
    return url


def _segment_id(scene_id: int, start_sec: float, duration_sec: float) -> str:
    return f"scene_{scene_id:05d}_t{int(round(start_sec)):05d}_{int(duration_sec)}s"


def _choose_segments(
    duration_sec: float,
    segment_duration: float,
    n_segments: int,
    seed: int,
) -> list[float]:
    """Return sorted list of non-overlapping start times within a video."""
    rng = random.Random(seed)
    max_start = duration_sec - segment_duration
    if max_start <= 0:
        return [0.0] if duration_sec >= segment_duration * 0.5 else []

    starts: list[float] = []
    attempts = 0
    while len(starts) < n_segments and attempts < n_segments * 10:
        t = rng.uniform(0, max_start)
        if not any(abs(t - s) < segment_duration for s in starts):
            starts.append(t)
        attempts += 1
    return sorted(starts)


def _choose_segments_binaware(
    scene_id: int,
    duration_sec: float,
    segment_duration: float,
    n_segments: int,
    interp_dir: Path,
    bins_def: list[dict] | None,
    blacklisted_bins: set[int],
    seed: int,
) -> list[float]:
    """Choose segments preferring underrepresented motion-magnitude bins.

    If an analysis JSON exists for this scene in interp_dir, filters candidate
    segments to only those whose bin_id is in the 3 least-populated bins
    (excluding blacklisted bins). Falls back to random selection if no analysis
    data exists or no valid candidates remain.
    """
    scene_name = f"scene_{scene_id:05d}"
    analysis_path = interp_dir / f"{scene_name}.json"
    rng = random.Random(seed)

    # Fall back to random if no bin-aware data available
    if bins_def is None or not analysis_path.exists():
        return _choose_segments(duration_sec, segment_duration, n_segments, seed)

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Find the 3 least populated bins (excluding blacklisted)
    eligible_bins = [b for b in bins_def if b["bin_id"] not in blacklisted_bins]
    if not eligible_bins:
        log.warning("All bins blacklisted, falling back to random selection")
        return _choose_segments(duration_sec, segment_duration, n_segments, seed)

    eligible_bins.sort(key=lambda b: b["count"])
    target_bin_ids = {b["bin_id"] for b in eligible_bins[:3]}

    # Filter analysis segments to those in target bins
    candidates = []
    for seg in analysis.get("segments", []):
        if seg["bin_id"] in target_bin_ids:
            candidates.append(seg)

    if not candidates:
        log.debug("No segments in target bins for %s, falling back to random", scene_name)
        return _choose_segments(duration_sec, segment_duration, n_segments, seed)

    # Shuffle and pick non-overlapping candidates
    rng.shuffle(candidates)
    starts: list[float] = []
    for cand in candidates:
        t = cand["start_time_sec"]
        # Ensure within video bounds
        if t + segment_duration > duration_sec:
            continue
        # Ensure non-overlapping
        if not any(abs(t - s) < segment_duration for s in starts):
            starts.append(t)
        if len(starts) >= n_segments:
            break

    return sorted(starts)


def _update_binsdef(binsdef_path: Path, bin_id: int) -> None:
    """Increment the count for a given bin in binsdef.json."""
    with open(binsdef_path) as f:
        bins_def = json.load(f)
    for b in bins_def:
        if b["bin_id"] == bin_id:
            b["count"] += 1
            break
    with open(binsdef_path, "w") as f:
        json.dump(bins_def, f, indent=2)


def _get_segment_bin(
    scene_id: int,
    start_sec: float,
    segment_duration: float,
    interp_dir: Path,
    bins_def: list[dict],
) -> int | None:
    """Look up which bin a segment belongs to from the analysis JSON."""
    scene_name = f"scene_{scene_id:05d}"
    analysis_path = interp_dir / f"{scene_name}.json"
    if not analysis_path.exists():
        return None

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Find the segment closest to start_sec
    best_seg = None
    best_dist = float("inf")
    for seg in analysis.get("segments", []):
        dist = abs(seg["start_time_sec"] - start_sec)
        if dist < best_dist:
            best_dist = dist
            best_seg = seg

    if best_seg is not None and best_dist < segment_duration:
        return best_seg["bin_id"]

    return None


def _try_lock(lock_dir: Path, scene_id: int) -> bool:
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"scene_{scene_id:05d}.lock"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


# ── Core: preprocess one segment ─────────────────────────────────────────────

def preprocess_segment(
    video_path: Path,
    output_path: Path,
    projection: str,
    target_height: int,
    eye: str,
    start_sec: float,
    duration_sec: float,
    half_rate: bool = False,
    ffmpeg_gpu: int | None = None,
) -> bool:
    """Run ffmpeg to extract and preprocess one video segment."""
    try:
        source_info = probe_video(video_path)
    except Exception as e:
        log.error("Cannot probe %s: %s", video_path.name, e)
        return False

    def _run(use_hw: bool) -> bool:
        cmd = build_preprocess_command(
            input_path=video_path,
            output_path=output_path,
            projection=projection,
            target_height=target_height,
            half_rate=half_rate,
            eye=eye,
            use_hw_accel=use_hw,
            source_codec=source_info["codec"],
            source_width=source_info["width"],
            source_height=source_info["height"],
            source_fps_expr=source_info.get("fps_expr"),
        )
        i_idx = cmd.index("-i")
        cmd.insert(i_idx, f"{start_sec:.2f}")
        cmd.insert(i_idx, "-ss")
        i_idx += 2
        cmd.insert(i_idx + 2, f"{duration_sec:.2f}")
        cmd.insert(i_idx + 2, "-t")

        env = os.environ.copy()
        if ffmpeg_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(ffmpeg_gpu)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("ffmpeg: %s", " ".join(str(c) for c in cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            if use_hw and "cuvid" in " ".join(cmd).lower():
                return None  # signal: retry
            log.error("ffmpeg failed: %s", (result.stderr or "")[-600:])
            return False
        return True

    r = _run(use_hw=True)
    if r is None:
        log.warning("HW decode failed, retrying with CPU")
        r = _run(use_hw=False)
    return bool(r)


# ── Core: extract labels ──────────────────────────────────────────────────────

def extract_labels(
    funscript_path: Path,
    output_dir: Path,
    seg_start_sec: float,
    seg_fps: float,
    seg_total_frames: int,
) -> bool:
    labels_path = output_dir / "labels.npy"
    if labels_path.exists():
        log.info("Labels already present for %s", output_dir.name)
        return True

    fs_data = load_funscript(funscript_path)
    actions = get_actions(fs_data)

    if seg_fps <= 0 or seg_total_frames <= 0:
        log.error(
            "Invalid segment timing for %s: fps=%s frames=%s",
            output_dir.name,
            seg_fps,
            seg_total_frames,
        )
        return False

    frame_timestamps_ms = (
        seg_start_sec + (np.arange(seg_total_frames, dtype=np.float64) / seg_fps)
    ) * 1000.0
    labels = actions_to_timestamps(actions, frame_timestamps_ms)

    labels = np.clip(labels, 0.0, 1.0)
    np.save(labels_path, labels)

    if labels.std() < 0.02:
        log.warning("LOW QUALITY: labels std=%.4f — segment may be in a flat region", labels.std())
    log.info("Labels: %d frames  range=[%.3f, %.3f]  std=%.3f",
             len(labels), labels.min(), labels.max(), labels.std())
    return True


# ── Reprocess labels mode ─────────────────────────────────────────────────────

import re as _re
_SEG_RE = _re.compile(r"^scene_(\d+)_t(\d+)_(\d+)s$")


def _parse_seg_name(seg_name: str) -> tuple[int, float, float] | None:
    """Parse scene_id, start_sec, duration_sec from a segment directory name.

    Returns None for legacy names without timing (e.g. 'scene_00018').
    """
    m = _SEG_RE.match(seg_name)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def _reprocess_all_labels(
    preprocessed_dir: Path,
    processed_dir: Path,
    xbvr_by_scene: dict[int, tuple[Path, Path]] | None = None,
) -> None:
    """Re-extract labels.npy for every segment found in the preprocessed folder.

    xbvr_by_scene: mapping of scene_id → (video_path, funscript_path) built from xbvr,
    used as a fallback when metadata.json lacks those fields (legacy format).
    """
    mp4_files = sorted(preprocessed_dir.glob("*.mp4"))
    if not mp4_files:
        log.info("No preprocessed segments found in %s", preprocessed_dir)
        return

    log.info("Reprocessing labels for %d preprocessed segments", len(mp4_files))
    ok_count = skipped = failed = 0

    for mp4_path in mp4_files:
        seg_name = mp4_path.stem
        scene_dir = processed_dir / seg_name
        meta_path = scene_dir / "metadata.json"

        if not meta_path.exists():
            log.warning("SKIP %s — no metadata.json in %s", seg_name, scene_dir)
            skipped += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        seg_fps: float = meta["fps"]
        seg_total_frames: int = meta["total_frames"]

        # ── Resolve timing ────────────────────────────────────────────────
        if "segment_start_sec" in meta and "segment_duration_sec" in meta:
            start_sec: float = meta["segment_start_sec"]
            duration_sec: float = meta["segment_duration_sec"]
        else:
            parsed = _parse_seg_name(seg_name)
            if parsed is None:
                log.warning("SKIP %s — legacy metadata with no parseable timing", seg_name)
                skipped += 1
                continue
            _, start_sec, duration_sec = parsed
            log.debug("Derived timing from name: start=%.0f  duration=%.0f", start_sec, duration_sec)

        # ── Resolve video + funscript paths ───────────────────────────────
        raw_video = meta.get("video_path", "")
        raw_script = meta.get("funscript_path", "")
        video_path: Path | None = None
        funscript_path: Path | None = None

        # Only trust video_path from metadata if it's NOT the preprocessed file itself
        if raw_video and "preprocessed" not in raw_video.replace("\\", "/"):
            video_path = Path(raw_video)
        if raw_script:
            funscript_path = Path(raw_script)

        # Fall back to xbvr lookup by scene_id
        if (video_path is None or funscript_path is None) and xbvr_by_scene is not None:
            parsed = _parse_seg_name(seg_name)
            if parsed is not None:
                scene_id = parsed[0]
                entry = xbvr_by_scene.get(scene_id)
                if entry:
                    if video_path is None:
                        video_path = entry[0]
                    if funscript_path is None:
                        funscript_path = entry[1]

        if video_path is None:
            log.warning("SKIP %s — cannot determine original video path", seg_name)
            skipped += 1
            continue
        if funscript_path is None:
            log.warning("SKIP %s — cannot determine funscript path", seg_name)
            skipped += 1
            continue

        if not video_path.exists():
            log.warning("SKIP %s — original video not found: %s", seg_name, video_path)
            skipped += 1
            continue
        if not funscript_path.exists():
            log.warning("SKIP %s — funscript not found: %s", seg_name, funscript_path)
            skipped += 1
            continue

        labels_path = scene_dir / "labels.npy"
        if labels_path.exists():
            labels_path.unlink()

        log.info("Reprocessing labels: %s", seg_name)
        ok = extract_labels(
            funscript_path=funscript_path,
            output_dir=scene_dir,
            seg_start_sec=start_sec,
            seg_fps=seg_fps,
            seg_total_frames=seg_total_frames,
        )
        if ok:
            ok_count += 1
        else:
            log.error("Label extraction failed for %s", seg_name)
            failed += 1

    log.info("═" * 50)
    log.info("Reprocess complete: %d ok, %d failed, %d skipped", ok_count, failed, skipped)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch import video segments + labels from xbvr library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",              default="configs/default.yaml")
    parser.add_argument("--data-dir",            default="data")
    parser.add_argument("--max-scenes",          type=int, default=50,
                        help="Max number of videos to sample from")
    parser.add_argument("--duration",            type=float, default=20.0,
                        help="Segment duration in seconds")
    parser.add_argument("--segments-per-video",  type=int, default=5,
                        help="Segments to sample per video")
    parser.add_argument("--projections",         nargs="+", default=["180_sbs"])
    parser.add_argument("--half-rate",          action="store_true",
                        help="Keep exact source timing but drop every other frame")
    parser.add_argument("--ffmpeg-gpu",          type=int, default=None,
                        help="GPU index for ffmpeg HW decode (e.g. 0)")
    parser.add_argument("--lock-dir",            default=None,
                        help="Lock directory for multi-machine coordination")
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--no-shuffle",          action="store_true",
                        help="Disable random shuffle of video order")
    parser.add_argument("--min-actions",         type=int, default=3,
                        help="Skip segments with fewer than N funscript actions")
    parser.add_argument("--overwrite",           action="store_true",
                        help="Re-import scenes that already exist")
    parser.add_argument("--reprocess-labels",    action="store_true",
                        help="Re-extract labels.npy for all segments found in preprocessed folder")
    parser.add_argument("--blacklist-bins",       type=str, default="",
                        help="Comma-separated bin IDs to exclude (e.g. '3,4,5')")
    args = parser.parse_args()

    # Parse blacklisted bins
    blacklisted_bins: set[int] = set()
    if args.blacklist_bins:
        blacklisted_bins = {int(b.strip()) for b in args.blacklist_bins.split(",") if b.strip()}
        log.info("Blacklisted bins: %s", blacklisted_bins)

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    data_dir = Path(args.data_dir)
    preprocessed_dir = data_dir / "preprocessed"
    processed_dir = data_dir / "processed"
    lock_dir = Path(args.lock_dir) if args.lock_dir else None
    interp_dir = data_dir / "interpolated"
    binsdef_path = data_dir / "binsdef.json"

    # ── Reprocess-labels mode ──────────────────────────────────────────────
    if args.reprocess_labels:
        xbvr_by_scene: dict[int, tuple[Path, Path]] | None = None
        try:
            db_url = _load_database_url()
            from src.data.xbvr import query_scene_pairs
            xbvr_pairs = query_scene_pairs(db_url, projections=None, min_duration=0.0)
            xbvr_by_scene = {p.scene_id: (p.video_path, p.funscript_path) for p in xbvr_pairs}
            log.info("Built xbvr lookup with %d entries", len(xbvr_by_scene))
        except Exception as e:
            log.warning("Could not load xbvr lookup (legacy-format segments may be skipped): %s", e)
        _reprocess_all_labels(
            preprocessed_dir=preprocessed_dir,
            processed_dir=processed_dir,
            xbvr_by_scene=xbvr_by_scene,
        )
        return

    # ── Load bin definitions (for bin-aware sampling) ─────────────────────
    bins_def: list[dict] | None = None
    if binsdef_path.exists():
        with open(binsdef_path) as f:
            bins_def = json.load(f)
        log.info("Loaded bin definitions: %d bins from %s", len(bins_def), binsdef_path)
    else:
        log.warning(
            "No binsdef.json found at %s — run analyze_funscripts.py first for "
            "bin-aware segment selection. Falling back to random selection.",
            binsdef_path,
        )

    # ── Discover videos from xbvr ─────────────────────────────────────────
    db_url = _load_database_url()
    pairs = list(pairs_from_xbvr(
        db_url, projections=args.projections,
        min_duration=args.duration * 1.5, check_exists=True,
    ))
    log.info("Found %d video+funscript pairs in library", len(pairs))

    rng = random.Random(args.seed)
    if not args.no_shuffle:
        rng.shuffle(pairs)

    # ── Find already-done scene IDs ───────────────────────────────────────
    done: set[str] = set()
    if processed_dir.exists():
        for d in processed_dir.iterdir():
            if d.is_dir() and (d / "labels.npy").exists() and (d / "review.json").exists():
                if not args.overwrite:
                    done.add(d.name)
    log.info("Already imported: %d segments", len(done))

    processed = skipped = failed = 0

    for pair in pairs[:args.max_scenes]:
        if not pair.video_path.exists():
            continue

        # Acquire lock (skip if another machine is already working this scene)
        if lock_dir and not _try_lock(lock_dir, pair.scene_id):
            log.info("Scene %d locked by another process, skipping", pair.scene_id)
            skipped += 1
            continue

        try:
            info = get_video_info(pair.video_path)
        except Exception:
            log.warning("Cannot read video info: %s", pair.video_path.name)
            continue

        try:
            fs_data = load_funscript(pair.funscript_path)
            actions = get_actions(fs_data)
        except Exception:
            log.warning("Cannot read funscript: %s", pair.funscript_path)
            continue

        action_times_sec = [a[0] / 1000.0 for a in actions]

        # ── Try bin-aware segment selection ────────────────────────────────
        starts = _choose_segments_binaware(
            pair.scene_id, info.duration_seconds, args.duration,
            args.segments_per_video, interp_dir, bins_def, blacklisted_bins,
            seed=args.seed + pair.scene_id,
        )

        for start_sec in starts:
            seg_name = _segment_id(pair.scene_id, start_sec, args.duration)

            # Skip if already fully imported
            if seg_name in done:
                log.debug("SKIP %s — already imported", seg_name)
                continue

            # Skip segments with insufficient funscript coverage
            n_actions = sum(1 for t in action_times_sec
                            if start_sec <= t <= start_sec + args.duration)
            if n_actions < args.min_actions:
                log.info("SKIP %s — only %d actions (need ≥%d)",
                         seg_name, n_actions, args.min_actions)
                continue

            preprocessed_path = preprocessed_dir / f"{seg_name}.mp4"
            scene_dir = processed_dir / seg_name
            scene_dir.mkdir(parents=True, exist_ok=True)

            log.info("─" * 50)
            log.info("Importing %s  (%s)", seg_name, pair.title)

            # ── Step 1: Preprocess video ──────────────────────────────────
            if not preprocessed_path.exists() or args.overwrite:
                projection = pair.video_projection or ("180_sbs" if cfg.video.vr_mode else "flat")
                ok = preprocess_segment(
                    video_path=pair.video_path,
                    output_path=preprocessed_path,
                    projection=projection,
                    target_height=cfg.video.frame_size,
                    eye=cfg.video.sbs_crop,
                    start_sec=start_sec,
                    duration_sec=args.duration,
                    half_rate=args.half_rate,
                    ffmpeg_gpu=args.ffmpeg_gpu,
                )
                if not ok:
                    log.error("ffmpeg failed for %s", seg_name)
                    failed += 1
                    continue
            else:
                log.info("Preprocessed video already present, skipping ffmpeg")

            # ── Step 2: Read output video info ────────────────────────────
            try:
                seg_info = get_video_info(preprocessed_path)
            except Exception as e:
                log.error("Cannot read output video: %s", e)
                failed += 1
                continue

            # ── Step 3: Save metadata ─────────────────────────────────────
            meta_path = scene_dir / "metadata.json"
            if not meta_path.exists() or args.overwrite:
                metadata = {
                    "video_path":         str(pair.video_path),
                    "funscript_path":     str(pair.funscript_path),
                    "segment_start_sec":  start_sec,
                    "segment_duration_sec": args.duration,
                    "source_fps":         info.fps,
                    "half_rate":          args.half_rate,
                    "width":              seg_info.width,
                    "height":             seg_info.height,
                    "fps":                seg_info.fps,
                    "total_frames":       seg_info.total_frames,
                    "duration_seconds":   seg_info.duration_seconds,
                }
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            # ── Step 4: Extract labels ────────────────────────────────────
            ok = extract_labels(
                funscript_path=pair.funscript_path,
                output_dir=scene_dir,
                seg_start_sec=start_sec,
                seg_fps=seg_info.fps,
                seg_total_frames=seg_info.total_frames,
            )
            if not ok:
                log.error("Label extraction failed for %s", seg_name)
                failed += 1
                continue

            # ── Step 5: Create review.json ────────────────────────────────
            review_path = scene_dir / "review.json"
            if not review_path.exists() or args.overwrite:
                write_review(scene_dir, {
                    "status":               "pending",
                    "stage2_status":        "pending",
                    "force_val":            False,
                    "imported_at":          datetime.now(timezone.utc).isoformat(),
                    "reviewed_at":          None,
                    "stage2_reviewed_at":   None,
                    "notes":                "",
                })

            log.info("Done: %s  →  open visualize_data.py to review", seg_name)
            processed += 1

            # ── Update bin counts ─────────────────────────────────────────
            if bins_def is not None:
                seg_bin = _get_segment_bin(
                    pair.scene_id, start_sec, args.duration, interp_dir, bins_def,
                )
                if seg_bin is not None:
                    _update_binsdef(binsdef_path, seg_bin)
                    # Refresh in-memory bins_def
                    with open(binsdef_path) as f:
                        bins_def = json.load(f)

    log.info("═" * 50)
    log.info("Complete: %d imported, %d failed, %d skipped (locked)", processed, failed, skipped)


if __name__ == "__main__":
    main()
