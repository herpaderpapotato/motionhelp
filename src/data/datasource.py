"""Unified data source interface for discovering video + funscript pairs.

Supports two modes:
1. Folder-based: Scan a directory for .mp4/.funscript pairs with matching names
2. XBVR database: Query the XBVR MySQL database for matched pairs

Both modes return the same ScenePair dataclass for downstream use.
"""

import logging
from pathlib import Path

from src.data.xbvr import ScenePair

log = logging.getLogger(__name__)


def pairs_from_folder(
    video_dir: str | Path,
    funscript_dir: str | Path | None = None,
    video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi"),
) -> list[ScenePair]:
    """Find video + funscript pairs by matching filenames in directories.

    Args:
        video_dir: Directory containing video files.
        funscript_dir: Directory containing .funscript files. If None, searches
            in the same directory as the videos.
        video_extensions: Video file extensions to look for.

    Returns:
        List of ScenePair objects for each matched pair found.
    """
    video_dir = Path(video_dir)
    funscript_dir = Path(funscript_dir) if funscript_dir else video_dir

    pairs: list[ScenePair] = []
    scene_id_counter = 0

    for ext in video_extensions:
        for video_path in sorted(video_dir.glob(f"*{ext}")):
            funscript_path = funscript_dir / f"{video_path.stem}.funscript"
            if funscript_path.exists():
                scene_id_counter += 1
                pairs.append(ScenePair(
                    scene_id=scene_id_counter,
                    title=video_path.stem,
                    video_path=video_path,
                    funscript_path=funscript_path,
                    video_width=0,  # unknown until probed
                    video_height=0,
                    video_projection="",
                    duration=0.0,
                ))

    log.info("Found %d video+funscript pairs in %s", len(pairs), video_dir)
    return pairs


def pairs_from_xbvr(
    database_url: str,
    projections: list[str] | None = None,
    min_duration: float = 60.0,
    check_exists: bool = True,
    limit: int | None = None,
) -> list[ScenePair]:
    """Get video + funscript pairs from the XBVR database.

    Args:
        database_url: MySQL connection string.
        projections: Filter to specific projection types (e.g. ["180_sbs"]).
        min_duration: Minimum video duration in seconds.
        check_exists: If True, verify files exist on disk.
        limit: Maximum number of pairs.

    Returns:
        List of ScenePair objects.
    """
    from src.data.xbvr import get_available_pairs, query_scene_pairs

    if check_exists:
        return get_available_pairs(database_url, projections, min_duration, limit)
    return query_scene_pairs(database_url, projections, min_duration, limit)
