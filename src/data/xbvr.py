"""XBVR database interface for discovering video + funscript pairs.

Queries the XBVR MySQL database to find scenes that have both video
and funscript files, returning their full file paths and metadata.
"""

import logging
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass
class ScenePair:
    """A matched video + funscript pair from the database."""
    scene_id: int
    title: str
    video_path: Path
    funscript_path: Path
    video_width: int
    video_height: int
    video_projection: str  # 180_sbs, fisheye190, mkx200, etc.
    duration: float  # seconds (from files table video_duration)


_PAIR_NAME_STOPWORDS = {
    "funscript",
    "script",
    "scripts",
    "video",
    "vr",
    "sbs",
    "tb",
    "left",
    "right",
    "180",
    "360",
    "flat",
    "mp4",
    "mkv",
    "avi",
    "mov",
}


def _pair_name_tokens(path: Path) -> list[str]:
    stem = re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).strip()
    return [token for token in stem.split() if token and token not in _PAIR_NAME_STOPWORDS]


def pair_name_similarity(video_path: Path, script_path: Path) -> float:
    """Return a soft similarity score for matching a script file to a video file."""
    video_tokens = _pair_name_tokens(video_path)
    script_tokens = _pair_name_tokens(script_path)
    if not video_tokens or not script_tokens:
        return 0.0

    video_norm = " ".join(video_tokens)
    script_norm = " ".join(script_tokens)
    shared = len(set(video_tokens) & set(script_tokens))
    union = len(set(video_tokens) | set(script_tokens))
    jaccard = shared / union if union else 0.0
    ratio = SequenceMatcher(None, video_norm, script_norm).ratio()
    contains = 1.0 if video_norm in script_norm or script_norm in video_norm else 0.0
    exact = 1.0 if video_path.stem.lower() == script_path.stem.lower() else 0.0
    return exact * 10.0 + contains * 5.0 + shared * 2.0 + jaccard + ratio


def _choose_best_scene_row(rows: list[tuple]) -> tuple:
    """Pick the most plausible video/script pairing from a scene's cross-joined rows."""
    return max(
        rows,
        key=lambda row: (
            pair_name_similarity(
                Path(row[2].replace("/", "\\")),
                Path(row[3].replace("/", "\\")),
            ),
            (row[4] or 0) * (row[5] or 0),
            row[7] or 0.0,
        ),
    )


def connect(database_url: str):
    """Connect to the XBVR MySQL database.

    Args:
        database_url: MySQL connection string from .env (e.g. mysql://user:pass@host:port/db)

    Returns:
        pymysql connection object
    """
    import pymysql

    url = database_url.replace("^&", "&")
    parsed = urlparse(url)
    return pymysql.connect(
        host=parsed.hostname,
        port=parsed.port or 3306,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path.lstrip("/"),
        charset="utf8mb4",
    )


def query_scene_pairs(
    database_url: str,
    projections: list[str] | None = None,
    min_duration: float = 60.0,
    limit: int | None = None,
) -> list[ScenePair]:
    """Query XBVR for scenes that have both a video file and a funscript.

    For scenes with multiple video or script files, scores the available
    cross-product rows by filename similarity and video resolution to choose the
    most plausible pairing.

    Args:
        database_url: MySQL connection string.
        projections: Filter to these projection types. None = all.
            Common values: ["180_sbs", "fisheye190", "mkx200"]
        min_duration: Minimum video duration in seconds.
        limit: Maximum number of pairs to return.

    Returns:
        List of ScenePair objects with resolved file paths.
    """
    conn = connect(database_url)
    cur = conn.cursor()

    try:
        # Get all video files for scripted scenes
        proj_filter = ""
        if projections:
            placeholders = ", ".join(["%s"] * len(projections))
            proj_filter = f"AND vf.video_projection IN ({placeholders})"

        query = f"""
            SELECT
                s.id AS scene_id,
                s.title,
                CONCAT(vf.path, '/', vf.filename) AS video_fullpath,
                CONCAT(sf.path, '/', sf.filename) AS script_fullpath,
                vf.video_width,
                vf.video_height,
                vf.video_projection,
                vf.video_duration
            FROM scenes s
            JOIN files vf ON vf.scene_id = s.id AND vf.type = 'video'
            JOIN files sf ON sf.scene_id = s.id AND sf.type = 'script'
            WHERE s.is_scripted = 1
                AND vf.video_width > 0
                AND vf.video_duration >= %s
                {proj_filter}
            ORDER BY s.id
        """

        params: list = [min_duration]
        if projections:
            params.extend(projections)

        cur.execute(query, params)
        rows = cur.fetchall()

        # Deduplicate per scene by picking the most plausible video/script pair.
        rows_by_scene: dict[int, list[tuple]] = {}
        for row in rows:
            rows_by_scene.setdefault(row[0], []).append(row)

        seen_scenes: dict[int, ScenePair] = {}
        for scene_id, scene_rows in rows_by_scene.items():
            row = _choose_best_scene_row(scene_rows)
            video_path = Path(row[2].replace("/", "\\"))
            script_path = Path(row[3].replace("/", "\\"))

            unique_videos = {scene_row[2] for scene_row in scene_rows}
            unique_scripts = {scene_row[3] for scene_row in scene_rows}
            if len(unique_videos) > 1 or len(unique_scripts) > 1:
                similarity = pair_name_similarity(video_path, script_path)
                log.debug(
                    "Scene %d candidate pairs=%d unique_videos=%d unique_scripts=%d chosen_similarity=%.3f",
                    scene_id,
                    len(scene_rows),
                    len(unique_videos),
                    len(unique_scripts),
                    similarity,
                )

            seen_scenes[scene_id] = ScenePair(
                scene_id=scene_id,
                title=row[1],
                video_path=video_path,
                funscript_path=script_path,
                video_width=row[4],
                video_height=row[5],
                video_projection=row[6] or "",
                duration=row[7] or 0.0,
            )

        pairs = list(seen_scenes.values())

        if limit:
            pairs = pairs[:limit]

        log.info("Found %d video+funscript pairs from XBVR database", len(pairs))
        return pairs

    finally:
        cur.close()
        conn.close()


def get_available_pairs(
    database_url: str,
    projections: list[str] | None = None,
    min_duration: float = 60.0,
    limit: int | None = None,
) -> list[ScenePair]:
    """Query pairs and filter to only those whose files actually exist on disk.

    Args:
        database_url: MySQL connection string.
        projections: Filter to these projection types.
        min_duration: Minimum video duration in seconds.
        limit: Maximum number of pairs to return.

    Returns:
        List of ScenePair objects where both files exist.
    """
    pairs = query_scene_pairs(database_url, projections, min_duration, limit=None)

    available = []
    missing_video = 0
    missing_script = 0

    for pair in pairs:
        if not pair.video_path.exists():
            missing_video += 1
            continue
        if not pair.funscript_path.exists():
            missing_script += 1
            continue
        available.append(pair)

    if missing_video or missing_script:
        log.info(
            "Filtered %d pairs: %d available, %d missing video, %d missing script",
            len(pairs), len(available), missing_video, missing_script,
        )

    if limit:
        available = available[:limit]

    return available
