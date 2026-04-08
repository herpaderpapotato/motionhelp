"""GPU-accelerated video decoding via torchcodec.

Provides chunked frame iteration with optional VR SBS crop, resize, and FPS
conversion — all on GPU. Falls back to ffmpeg subprocess or OpenCV CPU decode
when torchcodec CUDA is unavailable.

Output frames are [H, W, C] uint8 numpy arrays (for YOLO/RAFT compatibility)
or optionally [N, H, W, C] CUDA tensors for zero-copy GPU pipelines.
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _has_torchcodec_cuda() -> bool:
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


class GPUVideoDecoder:
    """Chunked GPU video decoder with VR and resize support.

    Usage:
        decoder = GPUVideoDecoder(
            "video.mp4", device="cuda",
            crop_left_half=True, target_size=640,
            target_fps=30.0, start_time=60.0, duration=40.0,
        )
        for chunk_tensor in decoder.iter_chunks(chunk_size=512):
            # chunk_tensor: [N, H, W, C] uint8 CUDA tensor
            process(chunk_tensor)
    """

    def __init__(
        self,
        video_path: str | Path,
        device: str = "cuda",
        crop_left_half: bool = False,
        target_size: int | None = 640,
        target_fps: float | None = 30.0,
        start_time: float | None = None,
        duration: float | None = None,
    ):
        from torchcodec.decoders import VideoDecoder

        self.video_path = Path(video_path)
        self.device = device
        self.crop_left_half = crop_left_half
        self.target_size = target_size
        self.target_fps = target_fps
        self.start_time = start_time
        self.duration = duration

        self._decoder = VideoDecoder(
            str(self.video_path),
            device=device,
            dimension_order="NHWC",
            seek_mode="approximate",
        )
        self.metadata = self._decoder.metadata
        self.source_fps = self.metadata.average_fps or 30.0
        self.num_source_frames = self.metadata.num_frames or len(self._decoder)

        self._frame_indices = self._compute_frame_indices()
        self.num_frames = len(self._frame_indices)

    def _compute_frame_indices(self) -> list[int]:
        """Compute which source frames to decode for the target FPS and time range."""
        src_fps = self.source_fps
        total_src = self.num_source_frames

        start_frame = 0
        if self.start_time is not None and self.start_time > 0:
            start_frame = int(self.start_time * src_fps)

        if self.duration is not None:
            end_frame = start_frame + int(self.duration * src_fps)
        else:
            end_frame = total_src

        start_frame = min(start_frame, total_src)
        end_frame = min(end_frame, total_src)

        if self.target_fps is not None and self.target_fps < src_fps:
            step = src_fps / self.target_fps
            indices = []
            pos = float(start_frame)
            while pos < end_frame:
                indices.append(int(round(pos)))
                pos += step
        else:
            indices = list(range(start_frame, end_frame))

        return indices

    def _process_chunk(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply crop and resize on GPU. Input/output: [N, H, W, C] uint8 CUDA."""
        if self.crop_left_half:
            half_w = frames.shape[2] // 2
            frames = frames[:, :, :half_w, :].contiguous()

        if self.target_size is not None:
            h, w = frames.shape[1], frames.shape[2]
            if h != self.target_size or w != self.target_size:
                # NCHW float for interpolate, then back to NHWC uint8
                nchw = frames.permute(0, 3, 1, 2).float()
                nchw = F.interpolate(
                    nchw,
                    size=(self.target_size, self.target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                frames = nchw.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)

        return frames

    def iter_chunks(
        self, chunk_size: int = 512, as_numpy: bool = False,
        decode_batch: int = 64,
    ):
        """Yield frame chunks as [N, H, W, C] uint8 tensors (CUDA) or numpy arrays.

        Args:
            chunk_size: Frames per output chunk.
            as_numpy: If True, yield numpy arrays on CPU. If False, yield CUDA tensors.
            decode_batch: Max frames to decode from source at once (prevents OOM
                         on high-res videos). Decoded frames are cropped/resized
                         before accumulating.
        """
        indices = self._frame_indices
        accumulated = []
        acc_count = 0

        for i in range(0, len(indices), decode_batch):
            batch_indices = indices[i : i + decode_batch]
            if not batch_indices:
                break

            frames = self._decoder.get_frames_at(batch_indices)
            data = frames.data if hasattr(frames, "data") else frames
            data = self._process_chunk(data)  # crop + resize → target_size

            accumulated.append(data)
            acc_count += len(data)

            # Yield full chunks
            while acc_count >= chunk_size:
                concat = torch.cat(accumulated, dim=0)
                yield_data = concat[:chunk_size]
                remainder = concat[chunk_size:]
                accumulated = [remainder] if remainder.shape[0] > 0 else []
                acc_count = remainder.shape[0] if remainder.shape[0] > 0 else 0

                if as_numpy:
                    yield yield_data.cpu().numpy()
                else:
                    yield yield_data

                del concat, yield_data

        # Yield remaining frames
        if accumulated:
            final = torch.cat(accumulated, dim=0)
            if final.shape[0] > 0:
                if as_numpy:
                    yield final.cpu().numpy()
                else:
                    yield final

    def iter_chunks_numpy(self, chunk_size: int = 512, decode_batch: int = 64):
        """Yield frame chunks as lists of numpy arrays (backward compatible)."""
        for chunk in self.iter_chunks(chunk_size, as_numpy=True, decode_batch=decode_batch):
            yield list(chunk)

    def close(self) -> None:
        self._decoder = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def stream_video_gpu(
    video_path: str | Path,
    device: str = "cuda",
    crop_left_half: bool = False,
    target_size: int | None = 640,
    target_fps: float | None = 30.0,
    start_time: float | None = None,
    duration: float | None = None,
    chunk_size: int = 512,
    as_numpy: bool = False,
):
    """Convenience generator: stream video frames in chunks via GPU decode.

    Falls back to ffmpeg/cv2 if torchcodec CUDA is not available.
    """
    if not _has_torchcodec_cuda():
        log.warning("torchcodec CUDA not available, falling back to ffmpeg/cv2")
        yield from _fallback_stream(
            video_path, target_size, crop_left_half,
            start_time, duration, target_fps, chunk_size,
        )
        return

    try:
        decoder = GPUVideoDecoder(
            video_path, device=device,
            crop_left_half=crop_left_half,
            target_size=target_size,
            target_fps=target_fps,
            start_time=start_time,
            duration=duration,
        )
        log.info(
            "GPU decode: %d frames @ %.1ffps from %s",
            decoder.num_frames, decoder.source_fps, video_path,
        )
        if as_numpy:
            yield from decoder.iter_chunks_numpy(chunk_size)
        else:
            yield from decoder.iter_chunks(chunk_size, as_numpy=False)
        decoder.close()
    except Exception as e:
        log.warning("torchcodec decode failed (%s), falling back to ffmpeg/cv2", e)
        # Release GPU memory before fallback
        torch.cuda.empty_cache()
        yield from _fallback_stream(
            video_path, target_size, crop_left_half,
            start_time, duration, target_fps, chunk_size,
        )


def _fallback_stream(
    video_path, target_size, crop_left_half,
    start_time, duration, target_fps, chunk_size,
):
    """Fallback: stream via OpenCV CPU decode."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_time is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    step = src_fps / target_fps if target_fps and target_fps < src_fps else 1.0
    max_frames = int(duration * target_fps) if duration and target_fps else float("inf")

    chunk: list[np.ndarray] = []
    frame_pos = 0.0
    next_target = 0.0
    total = 0

    while total < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_pos >= next_target:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_left_half:
                frame = frame[:, : frame.shape[1] // 2]
            if target_size is not None:
                frame = cv2.resize(frame, (target_size, target_size))
            chunk.append(frame)
            total += 1
            next_target += step

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        frame_pos += 1

    cap.release()
    if chunk:
        yield chunk
