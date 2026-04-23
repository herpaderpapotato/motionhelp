"""Scan processed clips for likely label or timing mismatches.

This is intended to rank suspicious clips before manual review. It compares the
stored labels against labels regenerated from the source funscript using the
current exact-timestamp sampling logic, and it can optionally estimate whether
the extracted clip starts at the expected source-video timestamp.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.curation import read_review
from src.data.funscript import actions_to_timestamps, get_actions, load_funscript
from src.data.preprocess import STEREO_LAYOUTS
from src.data.xbvr import pair_name_similarity, query_scene_pairs


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


def _load_database_url() -> str:
    from dotenv import dotenv_values

    env = dotenv_values(Path(__file__).parent.parent / ".env")
    return env.get("DATABASE_URL", "")


@dataclass
class MismatchFinding:
    scene: str
    status: str
    action_count: int
    stored_std: float
    stored_flat_frac: float
    regen_rmse: float
    regen_max_abs: float
    pair_similarity: float
    start_offset_sec: float | None
    start_match_score: float | None
    video_path: str
    funscript_path: str


def _segment_scene_id(scene_name: str) -> int | None:
    if not scene_name.startswith("scene_"):
        return None
    try:
        return int(scene_name.split("_")[1])
    except (IndexError, ValueError):
        return None


def _resolve_source_paths(
    scene_dir: Path,
    metadata: dict,
    xbvr_lookup: dict[int, tuple[Path, Path]] | None,
) -> tuple[Path | None, Path | None]:
    video_path = Path(metadata["video_path"]) if metadata.get("video_path") else None
    funscript_path = Path(metadata["funscript_path"]) if metadata.get("funscript_path") else None

    if video_path is not None and not video_path.exists():
        video_path = None
    if funscript_path is not None and not funscript_path.exists():
        funscript_path = None

    if xbvr_lookup is not None and (video_path is None or funscript_path is None):
        scene_id = _segment_scene_id(scene_dir.name)
        if scene_id is not None and scene_id in xbvr_lookup:
            xbvr_video, xbvr_script = xbvr_lookup[scene_id]
            video_path = video_path or xbvr_video
            funscript_path = funscript_path or xbvr_script

    return video_path, funscript_path


def _load_regenerated_labels(
    metadata: dict,
    actions: list[tuple[int, int]],
) -> np.ndarray:
    seg_fps = float(metadata["fps"])
    seg_total_frames = int(metadata["total_frames"])
    seg_start_sec = float(metadata["segment_start_sec"])
    timestamps_ms = (
        seg_start_sec + (np.arange(seg_total_frames, dtype=np.float64) / seg_fps)
    ) * 1000.0
    return actions_to_timestamps(actions, timestamps_ms)


def _preprocess_source_frame(
    frame_bgr: np.ndarray,
    projection: str,
    eye: str,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    layout = STEREO_LAYOUTS.get(projection or "", "sbs")
    h, w = frame_bgr.shape[:2]
    if layout == "sbs":
        half_w = w // 2
        frame_bgr = frame_bgr[:, :half_w] if eye == "left" else frame_bgr[:, half_w:]
    elif layout == "tb":
        half_h = h // 2
        frame_bgr = frame_bgr[:half_h, :] if eye == "left" else frame_bgr[half_h:, :]

    if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
        frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame_gray, (96, 96), interpolation=cv2.INTER_AREA)


def _estimate_start_offset(
    clip_path: Path,
    video_path: Path,
    metadata: dict,
    eye: str,
    search_window_sec: float,
) -> tuple[float | None, float | None]:
    clip_cap = cv2.VideoCapture(str(clip_path))
    if not clip_cap.isOpened():
        return None, None
    try:
        ok, clip_frame = clip_cap.read()
    finally:
        clip_cap.release()
    if not ok or clip_frame is None:
        return None, None

    source_cap = cv2.VideoCapture(str(video_path))
    if not source_cap.isOpened():
        return None, None

    try:
        source_fps = source_cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            return None, None

        expected_start = float(metadata["segment_start_sec"])
        projection = metadata.get("projection") or metadata.get("video_projection") or "180_sbs"
        target_width = int(metadata["width"])
        target_height = int(metadata["height"])
        target = _preprocess_source_frame(clip_frame, "mono", eye, 96, 96)

        max_frames = max(1, int(round(search_window_sec * source_fps)))
        expected_frame = int(round(expected_start * source_fps))

        best_offset_sec = None
        best_score = None
        for frame_offset in range(-max_frames, max_frames + 1):
            candidate_frame_idx = max(0, expected_frame + frame_offset)
            source_cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_frame_idx)
            ok, source_frame = source_cap.read()
            if not ok or source_frame is None:
                continue

            candidate = _preprocess_source_frame(
                source_frame,
                projection=projection,
                eye=eye,
                target_width=target_width,
                target_height=target_height,
            )
            score = float(np.mean(np.abs(candidate.astype(np.float32) - target.astype(np.float32))))
            if best_score is None or score < best_score:
                best_score = score
                best_offset_sec = frame_offset / source_fps

        return best_offset_sec, best_score
    finally:
        source_cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect processed clips for label or timing mismatches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--status", choices=["rejected", "pending", "approved", "all"], default="rejected")
    parser.add_argument("--max-scenes", type=int, default=200)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--check-video-start", action="store_true")
    parser.add_argument("--search-window-sec", type=float, default=2.0)
    parser.add_argument("--eye", choices=["left", "right"], default="left")
    parser.add_argument("--use-xbvr-fallback", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"

    xbvr_lookup: dict[int, tuple[Path, Path]] | None = None
    if args.use_xbvr_fallback:
        db_url = _load_database_url()
        if db_url:
            pairs = query_scene_pairs(db_url, projections=None, min_duration=0.0)
            xbvr_lookup = {pair.scene_id: (pair.video_path, pair.funscript_path) for pair in pairs}
            log.info("Loaded xbvr lookup with %d scene pairs", len(xbvr_lookup))

    findings: list[MismatchFinding] = []

    scene_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())

    for scene_dir in scene_dirs:
        metadata_path = scene_dir / "metadata.json"
        labels_path = scene_dir / "labels.npy"
        preprocessed_path = data_dir / "preprocessed" / f"{scene_dir.name}.mp4"
        if not metadata_path.exists() or not labels_path.exists():
            continue

        review = read_review(scene_dir)
        status = review.get("status", "pending")
        if args.status != "all" and status != args.status:
            continue
        if args.max_scenes > 0 and len(findings) >= args.max_scenes:
            break

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_path, funscript_path = _resolve_source_paths(scene_dir, metadata, xbvr_lookup)
        if funscript_path is None or not funscript_path.exists():
            continue

        try:
            stored_labels = np.load(labels_path)
            actions = get_actions(load_funscript(funscript_path))
            regenerated = _load_regenerated_labels(metadata, actions)
        except Exception as exc:
            log.debug("Skipping %s: %s", scene_dir.name, exc)
            continue

        if len(stored_labels) != len(regenerated):
            common = min(len(stored_labels), len(regenerated))
            stored_labels = stored_labels[:common]
            regenerated = regenerated[:common]

        flat_frac = float(np.mean((stored_labels <= 1e-4) | (stored_labels >= 1.0 - 1e-4))) if len(stored_labels) else 1.0
        regen_rmse = float(np.sqrt(np.mean((stored_labels - regenerated) ** 2))) if len(stored_labels) else 0.0
        regen_max_abs = float(np.max(np.abs(stored_labels - regenerated))) if len(stored_labels) else 0.0
        pair_similarity = pair_name_similarity(video_path, funscript_path) if video_path is not None else 0.0

        start_offset_sec = None
        start_match_score = None
        if args.check_video_start and video_path is not None and preprocessed_path.exists():
            start_offset_sec, start_match_score = _estimate_start_offset(
                clip_path=preprocessed_path,
                video_path=video_path,
                metadata=metadata,
                eye=args.eye,
                search_window_sec=args.search_window_sec,
            )

        findings.append(
            MismatchFinding(
                scene=scene_dir.name,
                status=status,
                action_count=len(actions),
                stored_std=float(stored_labels.std()) if len(stored_labels) else 0.0,
                stored_flat_frac=flat_frac,
                regen_rmse=regen_rmse,
                regen_max_abs=regen_max_abs,
                pair_similarity=pair_similarity,
                start_offset_sec=start_offset_sec,
                start_match_score=start_match_score,
                video_path=str(video_path) if video_path is not None else "",
                funscript_path=str(funscript_path),
            )
        )

    findings.sort(
        key=lambda finding: (
            abs(finding.start_offset_sec) if finding.start_offset_sec is not None else 0.0,
            finding.regen_rmse,
            1.0 - finding.pair_similarity,
            finding.stored_flat_frac,
        ),
        reverse=True,
    )

    for finding in findings[:args.top]:
        print(json.dumps(asdict(finding), ensure_ascii=True))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([asdict(finding) for finding in findings], indent=2),
            encoding="utf-8",
        )
        log.info("Wrote %d findings to %s", len(findings), out_path)


if __name__ == "__main__":
    main()