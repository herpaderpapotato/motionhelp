"""Extract spatial RoI features from preprocessed videos for DispositionTCN.

Runs YOLO inference on video frames and saves the full RoI-aligned spatial
feature grids [T, N, C, H, W] instead of mean-pooled embeddings [T, N, C].

Output per scene:
    data/processed/{scene_id}/spatial/{model_name}.npy       [T, N, C, roi_size, roi_size] float16
    data/processed/{scene_id}/spatial/{model_name}_conf.npy  [T, N] float32

Usage:
    python scripts/extract_spatial.py --n-train 30 --n-val 5
    python scripts/extract_spatial.py --n-train 50 --n-val 10 --roi-size 7
"""


import h5py
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.pose import load_pose_model
from src.data.extraction import (
    BACKBONE_STRIDE, FEATURE_LAYER_TYPES,
    PARTNER_CLASS, BEHOLDER_CLASS,
    _resolve_feature_layer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "vrlens-finetunes-multiclass-v2-yolo26m-pose"


class SpatialExtractor:
    """Extract spatial RoI feature grids (without mean-pooling).

    Like SinglePassExtractor but saves [n_det, C, roi_size, roi_size] instead
    of [n_det, C] per frame.
    """

    def __init__(
        self,
        model,
        roi_output_size: int = 7,
        confidence_threshold: float = 0.02,
        device: str = "cuda",
    ):
        self.model = model
        self.roi_output_size = roi_output_size
        self.max_persons = 1
        self.conf_threshold = confidence_threshold
        self.device = device
        self._features: torch.Tensor | None = None
        self.embed_dim: int | None = None

        self.layer_idx, self.layer_name = _resolve_feature_layer(model, None)
        self._hook = model.model.model[self.layer_idx].register_forward_hook(self._capture)
        log.info(
            "SpatialExtractor: hook on layer %d (%s), roi=%dx%d (combined box)",
            self.layer_idx, self.layer_name, roi_output_size, roi_output_size,
        )

    def _capture(self, module, input, output):
        self._features = output

    def extract_batch(
        self,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single combined spatial feature per frame.

        Computes one RoI-aligned grid [1, C, roi_size, roi_size] per frame
        using a bounding box that spans from the highest-confidence performer's
        top edge down to the bottom of the frame.  If no performer is detected
        a fallback box covering the centre-bottom half of the frame is used.

        Args:
            frames: [N, H, W, C] uint8 RGB numpy array.

        Returns:
            spatial: [N, 1, C, roi_size, roi_size] float16
            conf:    [N, 1] float32  (performer confidence, 0 if fallback)
        """
        n_frames = len(frames)
        R = self.roi_output_size
        frame_h, frame_w = frames.shape[1], frames.shape[2]

        self._features = None
        results = self.model.predict(
            list(frames),
            verbose=False,
            save=False,
            conf=self.conf_threshold,
            iou=0.97,
        )

        features = self._features
        if features is None:
            raise RuntimeError("No feature map captured")

        C = int(features.shape[1])
        self.embed_dim = C

        spatial_out = np.zeros((n_frames, 1, C, R, R), dtype=np.float16)
        conf_out = np.zeros((n_frames, 1), dtype=np.float32)

        for i, result in enumerate(results):
            # Find highest-confidence performer to determine box top/sides
            performer_box_xyxy = None
            performer_conf = 0.0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                det_conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                partner_idx = np.where(cls == PARTNER_CLASS)[0]
                if len(partner_idx) > 0:
                    best = partner_idx[int(np.argmax(det_conf[partner_idx]))]
                    performer_box_xyxy = boxes.xyxy[best]
                    performer_conf = float(det_conf[best])

            if performer_box_xyxy is not None:
                # Use performer left/right/top; extend to bottom of frame
                x1 = float(performer_box_xyxy[0])
                y1 = float(performer_box_xyxy[1])
                x2 = float(performer_box_xyxy[2])
                y2 = float(frame_h)
                conf_out[i, 0] = performer_conf
            else:
                # Fallback: centre-bottom half of frame
                x1 = 0.25 * frame_w
                y1 = 0.50 * frame_h
                x2 = 0.75 * frame_w
                y2 = float(frame_h)

            # Clamp to frame boundaries
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(frame_w), x2)
            y2 = min(float(frame_h), y2)

            box_tensor = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32, device=features.device
            )
            roi_feat = roi_align(
                features[i:i + 1],
                [box_tensor],
                output_size=R,
                spatial_scale=1.0 / BACKBONE_STRIDE,
                aligned=True,
            )  # [1, C, R, R]
            spatial_out[i, 0] = roi_feat[0].detach().cpu().to(torch.float16).numpy()

        return spatial_out, conf_out

    def close(self):
        if hasattr(self, "_hook"):
            self._hook.remove()
        self._features = None


def load_video_frames(video_path: Path, max_frames: int | None = None) -> np.ndarray:

    """Load video frames via torchcodec (GPU) or OpenCV fallback."""
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(video_path), device="cuda", dimension_order="NHWC")
        n = len(decoder) if max_frames is None else min(len(decoder), max_frames)
        frames = decoder.get_frames_in_range(0, n)
        return frames.data.cpu().numpy()
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return np.stack(frames)


def select_scenes(data_dir: Path, split: str, n: int) -> list[str]:
    """Select up to n scenes from a split that have preprocessed video."""
    split_file = data_dir / "splits" / f"{split}.json"
    with open(split_file) as f:
        all_ids = json.load(f)

    preprocessed = data_dir / "preprocessed"
    processed = data_dir / "processed"
    selected = []

    for vid_id in all_ids:
        if len(selected) >= n:
            break
        video_path = preprocessed / f"{vid_id}.mp4"
        label_path = processed / vid_id / "labels.npy"
        if not video_path.exists() or not label_path.exists():
            continue
        # Skip rejected scenes
        review_path = processed / vid_id / "review.json"
        if review_path.exists():
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("status") == "rejected" or review.get("stage2_status") == "rejected":
                continue
        selected.append(vid_id)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Extract spatial RoI features for DispositionTCN")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path = args.data_dir / "models" / "pose" / f"{args.model_name}.pt"
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)

    # Select scenes
    train_scenes = select_scenes(args.data_dir, "train", args.n_train)
    val_scenes = select_scenes(args.data_dir, "val", args.n_val)
    all_scenes = train_scenes + val_scenes
    log.info("Selected %d train + %d val = %d scenes", len(train_scenes), len(val_scenes), len(all_scenes))

    # Save scene lists for training
    spatial_meta_dir = args.data_dir / "splits"
    spatial_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(spatial_meta_dir / "disposition_train.json", "w") as f:
        json.dump(train_scenes, f, indent=2)
    with open(spatial_meta_dir / "disposition_val.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    log.info("Saved disposition splits to %s", spatial_meta_dir)

    # Load YOLO model
    log.info("Loading model: %s", model_path)
    pose_model = load_pose_model(
        model_name=args.model_name,
        model_path=str(model_path),
        device=args.device,
    )
    extractor = SpatialExtractor(
        pose_model,
        roi_output_size=args.roi_size,
        device=args.device,
    )

    preprocessed_dir = args.data_dir / "preprocessed"
    processed_dir = args.data_dir / "processed"
    t_total = time.perf_counter()

    for idx, scene_id in enumerate(all_scenes):
        split_label = "train" if scene_id in train_scenes else "val"
        log.info("[%d/%d] %s (%s)", idx + 1, len(all_scenes), scene_id, split_label)

        out_dir = processed_dir / scene_id / "spatial"
        # spatial_path = out_dir / f"{args.model_name}.npy"
        # conf_path = out_dir / f"{args.model_name}_conf.npy"
        spatial_path = out_dir / f"{args.model_name}.h5"
        conf_path = out_dir / f"{args.model_name}_conf.h5"

        if spatial_path.exists() and not args.overwrite:
            log.info("  SKIP — already extracted")
            continue

        video_path = preprocessed_dir / f"{scene_id}.mp4"
        n_frames = int(np.load(str(processed_dir / scene_id / "labels.npy"), mmap_mode="r").shape[0])

        t0 = time.perf_counter()
        frames = load_video_frames(video_path, max_frames=n_frames)
        actual_n = min(len(frames), n_frames)
        frames = frames[:actual_n]
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        all_spatial = []
        all_conf = []
        for start in range(0, actual_n, args.batch_size):
            batch = frames[start:start + args.batch_size]
            spatial, conf = extractor.extract_batch(batch)
            all_spatial.append(spatial)
            all_conf.append(conf)

        spatial_arr = np.concatenate(all_spatial, axis=0)
        conf_arr = np.concatenate(all_conf, axis=0)

        spatial_arr = spatial_arr.astype(np.float16)
        conf_arr = conf_arr.astype(np.float32)


        t_extract = time.perf_counter() - t0

        out_dir.mkdir(parents=True, exist_ok=True)
        # np.save(str(spatial_path), spatial_arr)
        # np.save(str(conf_path), conf_arr)
        with h5py.File(str(spatial_path), "w") as f:
            f.create_dataset("spatial", data=spatial_arr, compression="gzip")
        with h5py.File(str(conf_path), "w") as f:
            f.create_dataset("conf", data=conf_arr, compression="gzip")

        size_mb = spatial_arr.nbytes / 1e6
        log.info(
            "  Saved: %s shape=%s (%.0f MB) | load=%.1fs extract=%.1fs",
            spatial_path.name, spatial_arr.shape, size_mb, t_load, t_extract,
        )

    extractor.close()
    elapsed = time.perf_counter() - t_total
    log.info("Done: %d scenes in %.1fs", len(all_scenes), elapsed)


if __name__ == "__main__":
    main()
