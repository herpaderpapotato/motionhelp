"""Video loading and preprocessing utilities.

Handles VR side-by-side cropping, frame extraction, and resizing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str
    duration_seconds: float
    is_vr_sbs: bool = False  # Side-by-side VR detected


def get_video_info(path: str | Path) -> VideoInfo:
    """Get video metadata without loading frames."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 else 0

        # Heuristic: if width is roughly 2x height, it's likely SBS VR
        is_vr_sbs = width > height * 1.8

        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            codec=codec,
            duration_seconds=duration,
            is_vr_sbs=is_vr_sbs,
        )
    finally:
        cap.release()


class VideoReader:
    """Sequential video frame reader with VR preprocessing."""

    def __init__(
        self,
        path: str | Path,
        vr_mode: bool = True,
        sbs_crop: str = "left",
        target_size: int | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        self.path = Path(path)
        self.vr_mode = vr_mode
        self.sbs_crop = sbs_crop
        self.target_size = target_size
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.info = get_video_info(self.path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.path))
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        return self

    def __exit__(self, *args):
        if self._cap:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        if self._cap is None:
            raise RuntimeError("Use VideoReader as context manager: with VideoReader(...) as reader:")

        frame_idx = self.start_frame
        end = self.end_frame or self.info.total_frames

        while frame_idx < end:
            ret, frame = self._cap.read()
            if not ret:
                break

            frame = self._preprocess_frame(frame)
            yield frame_idx, frame
            frame_idx += 1

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """Read a specific frame by index."""
        if self._cap is None:
            raise RuntimeError("Use VideoReader as context manager")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        return self._preprocess_frame(frame)

    def read_batch(self, start: int, count: int) -> np.ndarray:
        """Read a batch of sequential frames. Returns [N, H, W, C] RGB array."""
        if self._cap is None:
            raise RuntimeError("Use VideoReader as context manager")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(count):
            ret, frame = self._cap.read()
            if not ret:
                break
            frames.append(self._preprocess_frame(frame))

        if not frames:
            return np.empty((0,), dtype=np.uint8)
        return np.stack(frames)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply VR crop and resize."""
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # VR SBS crop: take left or right half
        if self.vr_mode and self.info.is_vr_sbs:
            h, w = frame.shape[:2]
            half_w = w // 2
            if self.sbs_crop == "left":
                frame = frame[:, :half_w]
            else:
                frame = frame[:, half_w:]

        # Resize if target size specified
        if self.target_size is not None:
            frame = cv2.resize(frame, (self.target_size, self.target_size))

        return frame


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert [N, H, W, C] uint8 RGB numpy array to [N, C, H, W] float32 tensor in [0, 1]."""
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)  # [N, H, W, C] → [N, C, H, W]
    return t
