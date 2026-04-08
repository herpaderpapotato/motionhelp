"""Single-pass YOLO feature extraction: pose keypoints + RoI Align embeddings.

Extracts both per-frame pose keypoints AND per-person backbone embeddings in
a single YOLO forward pass by registering a forward hook on the backbone output
layer (C2PSA, layer 10). This eliminates the need for a separate model.embed()
call, saving ~11s per 1200-frame clip (~36% of total pipeline time).

Output:
    keypoints:  [N, max_persons, n_keypoints, 3]  (x_norm, y_norm, confidence)
    embeddings: [N, max_persons, 512]              (RoI-aligned backbone features)
"""

import logging
from typing import Any

import numpy as np
import torch
from torchvision.ops import roi_align

log = logging.getLogger(__name__)

BACKBONE_LAYER = 9   # SPPF (backbone output), stride 32, [B, 512, 20, 20] for 640px
BACKBONE_STRIDE = 32
EMBED_DIM = 512
ROI_OUTPUT_SIZE = 7


class SinglePassExtractor:
    """Extract pose keypoints and per-person embeddings in one YOLO forward pass.

    Registers a forward hook on the backbone output layer. During predict(),
    the hook captures the feature map, and RoI Align extracts per-person
    embeddings from detected bounding boxes.
    """

    def __init__(
        self,
        model: Any,
        layer_idx: int = BACKBONE_LAYER,
        max_persons: int = 10,
        n_keypoints: int = 21,
        confidence_threshold: float = 0.02,
        device: str = "cuda",
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.max_persons = max_persons
        self.n_keypoints = n_keypoints
        self.conf_threshold = confidence_threshold
        self.device = device
        self._features: torch.Tensor | None = None

        self._hook = model.model.model[layer_idx].register_forward_hook(self._capture)
        layer_name = type(model.model.model[layer_idx]).__name__
        log.info("Registered hook on layer %d (%s)", layer_idx, layer_name)

    def _capture(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._features = output

    def extract_batch(
        self,
        frames: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract keypoints and embeddings from a batch of frames.

        Args:
            frames: [N, H, W, C] uint8 RGB numpy array or torch tensor.

        Returns:
            keypoints:  [N, max_persons, n_keypoints, 3] float32
            embeddings: [N, max_persons, EMBED_DIM] float32
        """
        n_frames = len(frames)
        kp_out = np.zeros(
            (n_frames, self.max_persons, self.n_keypoints, 3), dtype=np.float32,
        )
        emb_out = np.zeros(
            (n_frames, self.max_persons, EMBED_DIM), dtype=np.float32,
        )

        if isinstance(frames, np.ndarray):
            predict_input = list(frames)
        else:
            predict_input = frames

        self._features = None
        results = self.model.predict(
            predict_input,
            verbose=False,
            save=False,
            conf=self.conf_threshold,
        )

        features = self._features  # [B, 512, H_feat, W_feat]

        for i, result in enumerate(results):
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            kpts = result.keypoints
            if not (hasattr(kpts, "xyn") and kpts.xyn is not None):
                continue

            kpts_data = kpts.xyn.cpu().numpy()  # [n_det, n_kpts, 2]
            conf_data = (
                kpts.conf.cpu().numpy()
                if kpts.conf is not None
                else np.ones(kpts_data.shape[:2])
            )  # [n_det, n_kpts]

            # Sort by detection confidence
            if result.boxes is not None and result.boxes.conf is not None:
                det_conf = result.boxes.conf.cpu().numpy()
                sorted_idx = np.argsort(-det_conf)
            else:
                sorted_idx = np.arange(len(kpts_data))

            n_persons = min(len(kpts_data), self.max_persons)
            for j in range(n_persons):
                idx = sorted_idx[j]
                kp_out[i, j, :, :2] = kpts_data[idx]
                kp_out[i, j, :, 2] = conf_data[idx]

            # RoI Align embeddings from backbone features
            if features is not None and result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy  # [n_det, 4] pixel coords
                selected_boxes = boxes_xyxy[sorted_idx[:n_persons]]  # [n_persons, 4]

                roi_features = roi_align(
                    features[i : i + 1],  # [1, 512, H_feat, W_feat]
                    [selected_boxes],
                    output_size=ROI_OUTPUT_SIZE,
                    spatial_scale=1.0 / BACKBONE_STRIDE,
                    aligned=True,
                )  # [n_persons, 512, roi_size, roi_size]

                embeddings = roi_features.mean(dim=[2, 3])  # [n_persons, 512]
                emb_out[i, :n_persons] = embeddings.detach().cpu().numpy()

        return kp_out, emb_out

    def close(self) -> None:
        if hasattr(self, "_hook"):
            self._hook.remove()
        self._features = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def extract_single_pass_batched(
    extractor: SinglePassExtractor,
    frames: list[np.ndarray] | np.ndarray,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract keypoints and embeddings in batches using SinglePassExtractor.

    Args:
        extractor: Initialized SinglePassExtractor.
        frames: List of [H, W, C] uint8 RGB frames.
        batch_size: Frames per inference batch.

    Returns:
        keypoints:  [N, max_persons, n_keypoints, 3] float32
        embeddings: [N, max_persons, EMBED_DIM] float32
    """
    n_frames = len(frames)
    all_kp = []
    all_emb = []

    for i in range(0, n_frames, batch_size):
        batch = (
            np.stack(frames[i : i + batch_size])
            if isinstance(frames, list)
            else frames[i : i + batch_size]
        )
        kp, emb = extractor.extract_batch(batch)
        all_kp.append(kp)
        all_emb.append(emb)

    return np.concatenate(all_kp, axis=0), np.concatenate(all_emb, axis=0)
