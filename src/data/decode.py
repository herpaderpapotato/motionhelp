"""GPU-accelerated video decoding via torchcodec.

Provides chunked frame iteration with optional VR SBS crop, resize, and exact
half-rate frame dropping — all on GPU. Falls back to ffmpeg subprocess or
OpenCV CPU decode when torchcodec CUDA is unavailable.

Output frames are [H, W, C] uint8 numpy arrays (for YOLO/RAFT compatibility)
or optionally [N, H, W, C] CUDA tensors for zero-copy GPU pipelines.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def effective_frame_rate(source_fps: float, half_rate: bool = False) -> float:
    """Return the effective output FPS for the selected decode mode."""
    if source_fps <= 0:
        return source_fps
    return source_fps / 2.0 if half_rate else source_fps


def compute_frame_sampling_plan(
    source_fps: float,
    num_source_frames: int | None,
    start_time: float | None = None,
    duration: float | None = None,
    half_rate: bool = False,
) -> tuple[int, int | None, int]:
    """Compute the frame range and stride for source-exact decode."""
    start_frame = 0
    if start_time is not None and start_time > 0:
        start_frame = max(0, int(start_time * source_fps))

    end_frame: int | None
    if duration is not None:
        end_frame = start_frame + int(duration * source_fps)
        if num_source_frames is not None:
            end_frame = min(end_frame, num_source_frames)
    else:
        end_frame = num_source_frames

    if num_source_frames is not None:
        start_frame = min(start_frame, num_source_frames)
        if end_frame is not None:
            end_frame = min(end_frame, num_source_frames)

    frame_step = 2 if half_rate else 1
    return start_frame, end_frame, frame_step


def estimate_output_frames(
    source_fps: float,
    num_source_frames: int | None,
    start_time: float | None = None,
    duration: float | None = None,
    half_rate: bool = False,
) -> int | None:
    """Estimate the number of decoded frames for the chosen sampling plan."""
    start_frame, end_frame, frame_step = compute_frame_sampling_plan(
        source_fps=source_fps,
        num_source_frames=num_source_frames,
        start_time=start_time,
        duration=duration,
        half_rate=half_rate,
    )
    if end_frame is None:
        return None
    if end_frame <= start_frame:
        return 0
    return (end_frame - start_frame + frame_step - 1) // frame_step


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
            half_rate=True, start_time=60.0, duration=40.0,
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
        half_rate: bool = False,
        start_time: float | None = None,
        duration: float | None = None,
    ):
        from torchcodec.decoders import VideoDecoder

        self.video_path = Path(video_path)
        self.device = device
        self.crop_left_half = crop_left_half
        self.target_size = target_size
        self.half_rate = half_rate
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
        self._decode_batch = self._auto_decode_batch()

    def _auto_decode_batch(self) -> int:
        """Compute decode_batch to keep peak VRAM under ~2 GB during crop+resize.

        The bottleneck is the float32 intermediate during F.interpolate:
          peak_bytes ≈ eff_w * eff_h * 3 * 4  (float32 per frame)
        For 8-K SBS this is ~199 MB/frame, so decode_batch=64 would need
        ~12.7 GB just for the resize step.  This limits it to ~10 frames.
        """
        try:
            src_w = getattr(self.metadata, "width", None)
            src_h = getattr(self.metadata, "height", None)
            if not src_w or not src_h:
                return 16  # conservative fallback
            eff_w = src_w // 2 if self.crop_left_half else src_w
            eff_h = src_h
            # float32 peak per frame in _process_chunk
            bytes_per_frame = eff_w * eff_h * 3 * 4
            target_bytes = 2 * 1024 ** 3  # 2 GB decode budget
            batch = max(1, min(64, int(target_bytes / bytes_per_frame)))
            log.info(
                "Auto decode_batch=%d for %dx%d source (eff %dx%d, %.0f MB/frame float32)",
                batch, src_w, src_h, eff_w, eff_h, bytes_per_frame / 1024 ** 2,
            )
            return batch
        except Exception:
            return 16

    def _compute_frame_indices(self) -> list[int]:
        """Compute which source frames to decode for the time range."""
        start_frame, end_frame, frame_step = compute_frame_sampling_plan(
            source_fps=self.source_fps,
            num_source_frames=self.num_source_frames,
            start_time=self.start_time,
            duration=self.duration,
            half_rate=self.half_rate,
        )
        if end_frame is None:
            end_frame = self.num_source_frames
        return list(range(start_frame, end_frame, frame_step))

    def _process_chunk(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply crop and resize on GPU. Input/output: [N, H, W, C] uint8 CUDA."""
        if self.crop_left_half:
            half_w = frames.shape[2] // 2
            cropped = frames[:, :, :half_w, :].contiguous()
            del frames  # free raw full-width tensor before proceeding
            frames = cropped

        if self.target_size is not None:
            h, w = frames.shape[1], frames.shape[2]
            if h != self.target_size or w != self.target_size:
                # NCHW float for interpolate, then back to NHWC uint8
                nchw = frames.permute(0, 3, 1, 2).float()
                del frames  # free uint8 while float32 is processed
                nchw = F.interpolate(
                    nchw,
                    size=(self.target_size, self.target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                result = nchw.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
                del nchw  # free large float32 once uint8 result is ready
                return result

        return frames

    def iter_chunks(
        self, chunk_size: int = 512, as_numpy: bool = False,
        decode_batch: int | None = None,
    ):
        """Yield frame chunks as [N, H, W, C] uint8 tensors (CUDA) or numpy arrays.

        Args:
            chunk_size: Frames per output chunk.
            as_numpy: If True, yield numpy arrays on CPU. If False, yield CUDA tensors.
            decode_batch: Max frames to decode from source at once (prevents OOM
                         on high-res videos). ``None`` uses the auto-computed value
                         (based on video resolution, targeting ~2 GB peak VRAM).
        """
        if decode_batch is None:
            decode_batch = self._decode_batch
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

    def iter_chunks_numpy(self, chunk_size: int = 512, decode_batch: int | None = None):
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
    half_rate: bool = False,
    start_time: float | None = None,
    duration: float | None = None,
    chunk_size: int = 512,
    as_numpy: bool = False,
    decode_batch: int | None = None,
):
    """Convenience generator: stream video frames in chunks via GPU decode.

    Falls back to ffmpeg/cv2 if torchcodec CUDA is not available.
    """
    if not _has_torchcodec_cuda():
        log.warning("torchcodec CUDA not available, falling back to ffmpeg/cv2")
        yield from _fallback_stream(
            video_path, target_size, crop_left_half,
            start_time, duration, half_rate, chunk_size,
        )
        return

    try:
        decoder = GPUVideoDecoder(
            video_path, device=device,
            crop_left_half=crop_left_half,
            target_size=target_size,
            half_rate=half_rate,
            start_time=start_time,
            duration=duration,
        )
        log.info(
            "GPU decode: %d frames @ %.3ffps effective (source %.3ffps) from %s",
            decoder.num_frames,
            effective_frame_rate(decoder.source_fps, decoder.half_rate),
            decoder.source_fps,
            video_path,
        )
        if as_numpy:
            yield from decoder.iter_chunks_numpy(chunk_size, decode_batch=decode_batch)
        else:
            yield from decoder.iter_chunks(chunk_size, as_numpy=False, decode_batch=decode_batch)
        decoder.close()
    except Exception as e:
        log.warning("torchcodec decode failed (%s), falling back to ffmpeg/cv2", e)
        # Release GPU memory before fallback
        torch.cuda.empty_cache()
        yield from _fallback_stream(
            video_path, target_size, crop_left_half,
            start_time, duration, half_rate, chunk_size,
        )


def _fallback_stream(
    video_path, target_size, crop_left_half,
    start_time, duration, half_rate, chunk_size,
):
    """Fallback: stream via OpenCV CPU decode."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    start_frame, end_frame, frame_step = compute_frame_sampling_plan(
        source_fps=src_fps,
        num_source_frames=total_src_frames,
        start_time=start_time,
        duration=duration,
        half_rate=half_rate,
    )
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    chunk: list[np.ndarray] = []
    source_frame_idx = start_frame

    while end_frame is None or source_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (source_frame_idx - start_frame) % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_left_half:
                frame = frame[:, : frame.shape[1] // 2]
            if target_size is not None:
                frame = cv2.resize(frame, (target_size, target_size))
            chunk.append(frame)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        source_frame_idx += 1

    cap.release()
    if chunk:
        yield chunk
