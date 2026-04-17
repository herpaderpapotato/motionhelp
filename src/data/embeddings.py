"""YOLO image embedding extraction via Ultralytics model.embed.

Extracts one 512-dim embedding per frame with ``model.embed(image)`` and stores
it in the first person slot. Additional person slots remain zero-filled so the
rest of the pipeline keeps the same ``[N_frames, max_persons, 512]`` layout.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_BACKBONE_LAYER = 10
EMBED_DIM = 512
EMBEDDING_FORMAT_VERSION = 4
EMBEDDING_METHOD = "single_pass_hook_roi_align"


def embeddings_metadata_path(embeddings_path: str | Path) -> Path:
    """Return the sidecar metadata path for an embeddings array."""
    return Path(embeddings_path).with_suffix(".json")


def load_embeddings_metadata(embeddings_path: str | Path) -> dict[str, Any] | None:
    """Load the sidecar metadata for an embeddings array if it exists."""
    meta_path = embeddings_metadata_path(embeddings_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def infer_embeddings_dim(embeddings_path: str | Path) -> int | None:
    """Infer embedding width from metadata first, then the array shape."""
    metadata = load_embeddings_metadata(embeddings_path)
    if metadata is not None and metadata.get("embed_dim") is not None:
        try:
            return int(metadata["embed_dim"])
        except (TypeError, ValueError):
            pass

    path = Path(embeddings_path)
    if not path.exists():
        return None

    try:
        embeddings = np.load(path, mmap_mode="r")
    except Exception:
        return None

    if embeddings.ndim != 3:
        return None
    return int(embeddings.shape[2])


def save_embeddings_artifacts(
    embeddings_path: str | Path,
    embeddings: np.ndarray,
    max_persons: int,
    video_path: str | Path | None = None,
) -> None:
    """Persist embeddings and sidecar metadata for cache validation."""
    out_path = Path(embeddings_path)
    np.save(out_path, embeddings)

    metadata: dict[str, Any] = {
        "format_version": EMBEDDING_FORMAT_VERSION,
        "method": EMBEDDING_METHOD,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "embed_dim": EMBED_DIM,
        "max_persons": max_persons,
    }
    if video_path is not None:
        metadata["video_path"] = str(video_path)

    embeddings_metadata_path(out_path).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def validate_embeddings_file(
    embeddings_path: str | Path,
    max_persons: int | None = None,
    embed_dim: int | None = None,
) -> tuple[bool, str]:
    """Validate that an embeddings cache matches the current extraction format."""
    path = Path(embeddings_path)
    if not path.exists():
        return False, "missing embeddings file"

    metadata = load_embeddings_metadata(path)
    if metadata is None:
        return False, "missing embeddings metadata"

    if metadata.get("format_version") != EMBEDDING_FORMAT_VERSION:
        return False, f"format_version={metadata.get('format_version')}"
    if metadata.get("method") != EMBEDDING_METHOD:
        return False, f"method={metadata.get('method')}"
    if embed_dim is not None and metadata.get("embed_dim") != embed_dim:
        return False, f"embed_dim={metadata.get('embed_dim')}"
    if max_persons is not None and int(metadata.get("max_persons", 0)) < max_persons:
        return False, f"max_persons={metadata.get('max_persons')}"

    try:
        embeddings = np.load(path, mmap_mode="r")
    except Exception as exc:
        return False, f"cannot load embeddings array: {exc}"

    if embeddings.ndim != 3:
        return False, f"expected 3 dims, got {embeddings.ndim}"
    if embed_dim is not None and embeddings.shape[2] != embed_dim:
        return False, f"expected embed_dim {embed_dim}, got {embeddings.shape[2]}"
    if max_persons is not None and embeddings.shape[1] < max_persons:
        return False, f"expected at least {max_persons} persons, got {embeddings.shape[1]}"

    return True, "ok"


class YOLOEmbeddingExtractor:
    """Extract one Ultralytics embedding vector per frame.

    The frame-level embedding is stored in slot 0 and remaining person slots
    stay zero-filled for compatibility with the existing model input layout.
    """

    def __init__(
        self,
        model: Any,
        layer_idx: int = DEFAULT_BACKBONE_LAYER,
        max_persons: int = 2,
        confidence_threshold: float = 0.3,
        device: str = "auto",
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.max_persons = max_persons
        self.conf_threshold = confidence_threshold
        self.device = device

    def extract_batch(self, frames: np.ndarray) -> np.ndarray:
        """Extract per-frame embeddings for a batch of frames.

        Args:
            frames: [N, H, W, C] uint8 RGB numpy array.

        Returns:
            [N, max_persons, EMBED_DIM] float32 numpy array.
            Slot 0 contains the frame embedding; remaining slots are zero-filled.
        """
        n_frames = len(frames)
        output = np.zeros((n_frames, self.max_persons, EMBED_DIM), dtype=np.float32)

        kwargs: dict[str, Any] = {"verbose": False}
        if self.device != "auto":
            kwargs["device"] = self.device

        predictor = getattr(self.model, "predictor", None)
        prev_embed = None
        had_prev_embed = False
        if predictor is not None and hasattr(predictor, "args"):
            had_prev_embed = hasattr(predictor.args, "embed")
            if had_prev_embed:
                prev_embed = predictor.args.embed

        try:
            embeddings = self.model.embed(list(frames), **kwargs)
        finally:
            # Ultralytics keeps predictor args between calls; restore embed so later
            # model.predict(...) returns Results instead of raw tensors.
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None and hasattr(predictor, "args"):
                if had_prev_embed:
                    predictor.args.embed = prev_embed
                else:
                    predictor.args.embed = None

        if len(embeddings) != n_frames:
            raise RuntimeError(
                f"model.embed returned {len(embeddings)} vectors for {n_frames} frames"
            )

        for i, embedding in enumerate(embeddings):
            if embedding is None:
                continue
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.as_tensor(embedding)
            embedding = embedding.detach().to(device="cpu", dtype=torch.float32).flatten()
            if embedding.numel() != EMBED_DIM:
                raise RuntimeError(
                    f"Expected embedding dim {EMBED_DIM}, got {tuple(embedding.shape)}"
                )
            output[i, 0] = embedding.numpy()

        return output

    def close(self) -> None:
        """Release extractor references."""
        self.model = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def extract_embeddings_video(
    model,
    video_path: str | Path,
    batch_size: int = 32,
    max_persons: int = 2,
    confidence_threshold: float = 0.3,
    layer_idx: int = DEFAULT_BACKBONE_LAYER,
    device: str = "auto",
) -> np.ndarray:
    """Extract YOLO embeddings for every frame in a video.

    Args:
        model: Loaded YOLO model (pose or detect variant).
        video_path: Path to preprocessed video (640×640, 10fps).
        batch_size: Frames per inference batch.
        max_persons: Number of person slots to reserve in the output.
        confidence_threshold: Unused, kept for call-site compatibility.
        layer_idx: Unused, kept for call-site compatibility.
        device: Device override for model.embed.

    Returns:
        [N_frames, max_persons, EMBED_DIM] float32 numpy array.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Extracting embeddings from %s (%d frames)", Path(video_path).name, total_frames)

    extractor = YOLOEmbeddingExtractor(
        model,
        layer_idx=layer_idx,
        max_persons=max_persons,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    all_embeddings = []
    batch_frames = []
    frames_read = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb)
            frames_read += 1

            if len(batch_frames) == batch_size:
                batch_arr = np.stack(batch_frames)
                batch_emb = extractor.extract_batch(batch_arr)
                all_embeddings.append(batch_emb)
                batch_frames = []

                if frames_read % (batch_size * 10) == 0:
                    log.info("  %d / %d frames", frames_read, total_frames)

        # Process remaining frames
        if batch_frames:
            batch_arr = np.stack(batch_frames)
            batch_emb = extractor.extract_batch(batch_arr)
            all_embeddings.append(batch_emb)

    finally:
        cap.release()
        extractor.close()

    if not all_embeddings:
        return np.zeros((0, max_persons, EMBED_DIM), dtype=np.float32)

    result = np.concatenate(all_embeddings, axis=0)  # [N, max_persons, 512]
    log.info("Extracted embeddings: %s (%.1f MB)", result.shape, result.nbytes / 1e6)
    return result
