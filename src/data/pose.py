"""YOLO pose extraction from video frames.

Wraps Ultralytics YOLO pose models to extract pose keypoint data.
Output: [N_frames, max_persons, n_keypoints, 3] where 3 = (x_norm, y_norm, confidence)

n_keypoints is detected automatically from the model (17 for standard COCO models,
21 for custom VR-finetuned models with pelvis/umbilicus/sternum keypoints).
"""

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

# COCO 17 keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_pose_model(
    model_name: str = "yolo11m-pose",
    model_path: str = "",
    device: str = "auto",
):
    """Load a YOLO pose model.

    Args:
        model_name: YOLO model variant (e.g., "yolo11m-pose").
        model_path: Path to custom model weights. If empty, downloads default.
        device: "auto", "cuda", or "cpu".
    """
    from ultralytics import YOLO

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
    else:
        model = YOLO(model_name)

    model.to(device)
    log.info("Loaded pose model %s on %s", model_name, device)
    return model


def extract_pose_batch(
    model,
    frames: np.ndarray | torch.Tensor,
    max_persons: int = 10,
    confidence_threshold: float = 0.02,
    n_keypoints: int | None = None,
) -> np.ndarray:
    """Extract pose keypoints from a batch of frames.

    Args:
        model: Loaded YOLO pose model.
        frames: [N, H, W, C] uint8 RGB numpy array, or [N, C, H, W] float tensor.
        max_persons: Maximum persons to track per frame.
        confidence_threshold: Minimum detection confidence.
        n_keypoints: Number of keypoints expected from the model. If None,
            auto-detected from the first result (17 for COCO, 21 for custom).

    Returns:
        np.ndarray of shape [N, max_persons, n_keypoints, 3] with normalized (x, y, conf).
    """
    results = model.predict(
        list(frames) if isinstance(frames, np.ndarray) and frames.ndim == 4 else frames,
        verbose=False,
        save=False,
        conf=confidence_threshold,
        embed=None,
    )

    n_frames = len(results)

    # Detect keypoint count from first result if not specified
    detected_n_kpts = n_keypoints
    if detected_n_kpts is None:
        detected_n_kpts = 17  # COCO default
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts = result.keypoints
                if hasattr(kpts, 'xyn') and kpts.xyn is not None and kpts.xyn.shape[0] > 0:
                    detected_n_kpts = kpts.xyn.shape[1]
                    break

    output = np.zeros((n_frames, max_persons, detected_n_kpts, 3), dtype=np.float32)

    for i, result in enumerate(results):
        if result.keypoints is None or len(result.keypoints) == 0:
            continue

        kpts = result.keypoints
        # kpts.xyn gives normalized coordinates [0,1]
        if hasattr(kpts, 'xyn') and kpts.xyn is not None:
            kpts_data = kpts.xyn.cpu().numpy()  # [n_detections, n_kpts, 2]
            conf_data = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones((*kpts_data.shape[:2],))  # [n_detections, n_kpts]
        else:
            continue

        # Sort by detection confidence (highest first)
        if result.boxes is not None and result.boxes.conf is not None:
            det_conf = result.boxes.conf.cpu().numpy()
            sorted_idx = np.argsort(-det_conf)
        else:
            sorted_idx = np.arange(len(kpts_data))

        n_persons = min(len(kpts_data), max_persons)
        for j in range(n_persons):
            idx = sorted_idx[j]
            output[i, j, :, :2] = kpts_data[idx]  # x, y normalized
            output[i, j, :, 2] = conf_data[idx]    # confidence

    return output


def extract_pose_video(
    model,
    video_path: str | Path,
    vr_mode: bool = True,
    sbs_crop: str = "left",
    frame_size: int = 640,
    batch_size: int = 32,
    max_persons: int = 10,
    confidence_threshold: float = 0.02,
    start_frame: int = 0,
    end_frame: int | None = None,
    n_keypoints: int | None = None,
) -> np.ndarray:
    """Extract pose keypoints from an entire video.

    Returns:
        np.ndarray of shape [N_frames, max_persons, n_keypoints, 3].
        n_keypoints is auto-detected from the model if not provided.
    """
    from .video import VideoReader

    with VideoReader(
        video_path,
        vr_mode=vr_mode,
        sbs_crop=sbs_crop,
        target_size=frame_size,
        start_frame=start_frame,
        end_frame=end_frame,
    ) as reader:
        all_keypoints = []
        batch_frames = []
        frame_count = 0

        for _idx, frame in reader:
            batch_frames.append(frame)

            if len(batch_frames) >= batch_size:
                batch_array = np.stack(batch_frames)
                kpts = extract_pose_batch(model, batch_array, max_persons, confidence_threshold, n_keypoints)
                if n_keypoints is None and kpts.ndim == 4:
                    n_keypoints = kpts.shape[2]  # detect from first batch
                all_keypoints.append(kpts)
                frame_count += len(batch_frames)
                batch_frames = []

                if frame_count % (batch_size * 10) == 0:
                    log.info("Processed %d frames", frame_count)

        # Process remaining frames
        if batch_frames:
            batch_array = np.stack(batch_frames)
            kpts = extract_pose_batch(model, batch_array, max_persons, confidence_threshold, n_keypoints)
            if n_keypoints is None and kpts.ndim == 4:
                n_keypoints = kpts.shape[2]
            all_keypoints.append(kpts)
            frame_count += len(batch_frames)

    if not all_keypoints:
        return np.empty((0, max_persons, n_keypoints if n_keypoints is not None else 17, 3), dtype=np.float32)

    result = np.concatenate(all_keypoints, axis=0)
    log.info("Extracted pose from %d frames, shape %s", frame_count, result.shape)
    return result
