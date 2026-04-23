"""Background spatial extractor for DispositionTCN.

Runs YOLO inference on video frames to capture neck feature maps via forward
hooks, then extracts multi-scale RoI features [T, N, C, H, W] using a fixed
centre-bottom region of the frame. The default output is a quantised
single-file HDF5 cache to keep training I/O cost close to the original
single-scale pipeline.

RoI region: [W*0.25, H*0.50, W*0.75, H] — covers the centre-bottom half of
every frame unconditionally.  Detection boxes are not used for RoI placement,
so /conf is always zero.

Output per scene:
    data/processed/{scene_id}/spatial/{model_name}.h5
        /spatial  [T, N, C, roi_size, roi_size] int8 or float16
        /conf     [T, N] float32  (always 0.0 with fixed-region extraction)

Usage:
    python scripts/extract_spatial.py
    python scripts/extract_spatial.py --watch
    python scripts/extract_spatial.py --scenes scene_00072_t01094_8s
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.curation import discover_scenes, inspect_model_embeddings_path
from src.data.pose import load_pose_model
from src.data.spatial import (
    build_channel_slices,
    resolve_disposition_feature_layers,
    save_spatial_features_h5,
    spatial_feature_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "vrlens-finetunes-multiclass-v2-yolo26m-pose"


class SpatialExtractor:
    """Extract fused multi-scale spatial RoI feature grids.

    Uses a fixed centre-bottom RoI (W*0.25–W*0.75, H*0.50–H) for every frame.
    YOLO inference is still run in full so that the neck feature maps are
    captured via forward hooks; detection boxes are not used.
    """

    def __init__(
        self,
        model,
        roi_output_size: int = 7,
        confidence_threshold: float = 0.02,
        device: str = "cuda",
        scale_layer_indices: tuple[int, ...] | None = None,
        scale_strides: tuple[int, ...] | None = None,
    ):
        self.model = model
        self.roi_output_size = roi_output_size
        self.max_persons = 1
        self.conf_threshold = confidence_threshold
        self.device = device
        self._features: dict[str, torch.Tensor | None] = {}
        self.embed_dim: int | None = None
        self.channel_slices: dict[str, list[int]] | None = None
        self.scale_specs = resolve_disposition_feature_layers(
            model,
            layer_indices=scale_layer_indices,
            strides=scale_strides,
        )
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        for spec in self.scale_specs:
            hook = model.model.model[spec["layer_idx"]].register_forward_hook(
                self._make_capture(spec["name"])
            )
            self._hooks.append(hook)
        log.info(
            "SpatialExtractor: hooks on %s, roi=%dx%d (combined box)",
            ", ".join(
                f"{spec['name']}=layer {spec['layer_idx']} ({spec['layer_name']}, stride {spec['stride']})"
                for spec in self.scale_specs
            ),
            roi_output_size,
            roi_output_size,
        )

    def _make_capture(self, scale_name: str):
        def _capture(module, input, output):
            self._features[scale_name] = output

        return _capture

    def export_metadata(self) -> dict[str, object]:
        return {
            "scale_specs": self.scale_specs,
            "channel_slices": self.channel_slices,
            "source_layers": [spec["layer_idx"] for spec in self.scale_specs],
            "source_strides": [spec["stride"] for spec in self.scale_specs],
        }

    def extract_batch(
        self,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single combined spatial feature per frame.

        Runs YOLO inference on the batch to fire the neck feature-map hooks,
        then applies RoI-align with a fixed centre-bottom box
        [W*0.25, H*0.50, W*0.75, H] for every frame.

        Args:
            frames: [N, H, W, C] uint8 RGB numpy array.

        Returns:
            spatial: [N, 1, C, roi_size, roi_size] float16
            conf:    [N, 1] float32 (always 0.0; reserved for future use)
        """
        n_frames = len(frames)
        roi_size = self.roi_output_size
        frame_h, frame_w = frames.shape[1], frames.shape[2]

        self._features = {spec["name"]: None for spec in self.scale_specs}
        results = self.model.predict(
            list(frames),
            verbose=False,
            save=False,
            conf=self.conf_threshold,
            iou=0.97,
        )

        features_by_name = {
            spec["name"]: self._features.get(spec["name"])
            for spec in self.scale_specs
        }
        missing = [
            spec for spec in self.scale_specs if features_by_name[spec["name"]] is None
        ]
        if missing:
            raise RuntimeError(
                "Missing feature maps for scales: "
                + ", ".join(spec["name"] for spec in missing)
            )

        channel_counts = [
            int(features_by_name[spec["name"]].shape[1])
            for spec in self.scale_specs
        ]
        self.channel_slices = build_channel_slices(self.scale_specs, channel_counts)
        channels = int(sum(channel_counts))
        self.embed_dim = channels

        spatial_out = np.zeros((n_frames, 1, channels, roi_size, roi_size), dtype=np.float16)
        conf_out = np.zeros((n_frames, 1), dtype=np.float32)

        for i in range(n_frames):
            x1 = 0.25 * frame_w
            y1 = 0.50 * frame_h
            x2 = 0.75 * frame_w
            y2 = float(frame_h)

            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(frame_w), x2)
            y2 = min(float(frame_h), y2)

            box_tensor = torch.tensor(
                [[x1, y1, x2, y2]],
                dtype=torch.float32,
                device=next(iter(features_by_name.values())).device,
            )
            roi_chunks = []
            for spec in self.scale_specs:
                scale_features = features_by_name[spec["name"]]
                roi_feat = roi_align(
                    scale_features[i:i + 1],
                    [box_tensor],
                    output_size=roi_size,
                    spatial_scale=1.0 / float(spec["stride"]),
                    aligned=True,
                )
                roi_chunks.append(roi_feat[0])
            fused_roi = torch.cat(roi_chunks, dim=0)
            spatial_out[i, 0] = fused_roi.detach().cpu().to(torch.float16).numpy()

        return spatial_out, conf_out

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features = {}


def load_video_frames(video_path: Path, max_frames: int | None = None) -> np.ndarray:
    """Load video frames via torchcodec (GPU) or OpenCV fallback."""
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(video_path), device="cuda", dimension_order="NHWC")
        n_frames = len(decoder) if max_frames is None else min(len(decoder), max_frames)
        frames = decoder.get_frames_in_range(0, n_frames)
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
        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        return np.stack(frames)


def process_scene(
    scene_dir: Path,
    preprocessed_dir: Path,
    extractor: SpatialExtractor,
    *,
    model_name: str,
    batch_size: int,
    storage_dtype: str,
    compression: str,
    overwrite: bool = False,
) -> bool:
    """Extract spatial features for one scene. Returns True on success."""
    scene_id = scene_dir.name
    video_path = preprocessed_dir / f"{scene_id}.mp4"
    if not video_path.exists():
        log.warning("SKIP %s - no preprocessed video at %s", scene_id, video_path)
        return False

    # emb_path, emb_reason = inspect_model_embeddings_path(scene_dir, model_name)
    # if emb_path is None:
    #     log.warning("SKIP %s - %s", scene_id, emb_reason)
    #     return False

    spatial_path = spatial_feature_path(scene_dir, model_name)
    if spatial_path.exists() and not overwrite:
        log.info("SKIP %s - spatial already exists at %s", scene_id, spatial_path)
        return True

    labels_path = scene_dir / "labels.npy"
    if not labels_path.exists():
        log.warning("SKIP %s - missing labels.npy", scene_id)
        return False

    n_frames = int(np.load(str(labels_path), mmap_mode="r").shape[0])
    log.info("Extracting spatial: %s (%d frames)", scene_id, n_frames)

    t0 = time.perf_counter()
    frames = load_video_frames(video_path, max_frames=n_frames)
    actual_n = min(len(frames), n_frames)
    if actual_n == 0:
        log.warning("SKIP %s - video decoder produced 0 frames", scene_id)
        return False
    frames = frames[:actual_n]
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_spatial = []
    all_conf = []
    for start in range(0, actual_n, batch_size):
        batch = frames[start:start + batch_size]
        spatial, conf = extractor.extract_batch(batch)
        all_spatial.append(spatial)
        all_conf.append(conf)

    spatial_arr = np.concatenate(all_spatial, axis=0).astype(np.float16)
    conf_arr = np.concatenate(all_conf, axis=0).astype(np.float32)
    t_extract = time.perf_counter() - t0

    if actual_n != n_frames:
        log.warning("Frame count mismatch for %s: labels=%d decoded=%d", scene_id, n_frames, actual_n)

    metadata = extractor.export_metadata()
    metadata.update(
        {
            "model_name": model_name,
            "n_frames": int(spatial_arr.shape[0]),
            #"embeddings_path": str(emb_path),
        }
    )
    save_spatial_features_h5(
        spatial_path,
        spatial_arr,
        conf_arr,
        storage_dtype=storage_dtype,
        compression=compression,
        metadata=metadata,
    )

    raw_mb = spatial_arr.nbytes / 1e6
    stored_mb = spatial_path.stat().st_size / 1e6
    log.info(
        "Saved spatial: %s  shape=%s raw=%.0f MB stored=%.0f MB | load=%.1fs extract=%.1fs",
        spatial_path,
        spatial_arr.shape,
        raw_mb,
        stored_mb,
        t_load,
        t_extract,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Background spatial extractor for DispositionTCN")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running, re-scan on --poll-interval",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between scans in watch mode (default: 30)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Only process these scene IDs (default: all pending)",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        choices=["int8", "float16"],
        default="int8",
        help="On-disk dtype for the spatial tensor; int8 uses per-channel quantisation",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression for the spatial cache",
    )
    args = parser.parse_args()




    model_path = args.data_dir / "models" / "pose" / f"{args.model_name}.pt"
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        if args.model_name == DEFAULT_MODEL:
            # downloading from Hugging Face https://huggingface.co/herpaderpapotato/pose-vrlens-finetunes-multiclass-v2-26/resolve/main/yolo26m-pose/weights/best.pt
            # to file DEFAULT_MODEL.pt
            import requests
            url = "https://huggingface.co/herpaderpapotato/pose-vrlens-finetunes-multiclass-v2-26/resolve/main/yolo26m-pose/weights/best.pt"
            log.info("Attempting to download default model from %s", url)
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                log.info("Downloaded model to %s", model_path)
            except Exception as e:
                log.error("Failed to download model: %s", e)
                sys.exit(1)

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

    def run_once() -> None:
        all_scenes = discover_scenes(args.data_dir, include_rejected=False, require_labels=True)
        pending = []
        for scene_id, state in all_scenes:
            if state.get("status") == "rejected" or state.get("stage2_status") == "rejected":
                continue
            if args.scenes and scene_id not in args.scenes:
                continue

            scene_dir = processed_dir / scene_id
            video_path = preprocessed_dir / f"{scene_id}.mp4"
            if not video_path.exists():
                continue
            # if inspect_model_embeddings_path(scene_dir, args.model_name)[0] is None:
            #     continue
            if not args.overwrite and spatial_feature_path(scene_dir, args.model_name).exists():
                continue
            pending.append(scene_id)

        log.info("Found %d scenes needing spatial features", len(pending))
        ok = failed = 0
        for scene_id in pending:
            try:
                success = process_scene(
                    processed_dir / scene_id,
                    preprocessed_dir,
                    extractor,
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    storage_dtype=args.storage_dtype,
                    compression=args.compression,
                    overwrite=args.overwrite,
                )
                if success:
                    ok += 1
                else:
                    failed += 1
            except Exception:
                log.exception("Failed: %s", scene_id)
                failed += 1
        log.info("Batch complete: %d ok, %d failed", ok, failed)

    try:
        if args.watch:
            log.info("Watch mode (poll every %ds) - Ctrl+C to stop", args.poll_interval)
            while True:
                run_once()
                time.sleep(args.poll_interval)
        else:
            run_once()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()