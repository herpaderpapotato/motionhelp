"""Single-pass YOLO feature extraction: pose keypoints + RoI Align embeddings.

Supports both single-class and multiclass models. Multiclass mode separates
detections by class (e.g. partner vs beholder) and stores them in separate
output slots.

Output (single-class):
    keypoints:  [N, max_persons, n_keypoints, 3]  (x_norm, y_norm, confidence)
    embeddings: [N, max_persons, 512]              (RoI-aligned backbone features)

Output (multiclass):
    partner_kp:   [N, max_partners, 21, 3]
    partner_emb:  [N, max_partners, 512]
    beholder_kp:  [N, max_beholders, 7, 3]   (only real keypoints 0-6)
    beholder_emb: [N, max_beholders, 512]
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

# Multiclass constants
PARTNER_CLASS = 0
BEHOLDER_CLASS = 1
PARTNER_N_KEYPOINTS = 21
BEHOLDER_N_KEYPOINTS = 7  # real keypoints at indices 0-6 of the 21 output


class SinglePassExtractor:
    """Extract pose keypoints and per-person embeddings in one YOLO forward pass.

    Registers a forward hook on the backbone output layer. During predict(),
    the hook captures the feature map, and RoI Align extracts per-person
    embeddings from detected bounding boxes.

    Multiclass mode (multiclass=True) separates detections by class:
      - Slots 0..max_partners-1  → partner detections (class 0)
      - Slots max_partners..max_partners+max_beholders-1 → beholder detections (class 1)
      - Beholder keypoints are truncated to indices 0..n_beholder_keypoints-1
    """

    def __init__(
        self,
        model: Any,
        layer_idx: int = BACKBONE_LAYER,
        max_persons: int = 10,
        n_keypoints: int = 21,
        confidence_threshold: float = 0.001,
        device: str = "cuda",
        # Multiclass options
        multiclass: bool = False,
        max_partners: int = 5,
        max_beholders: int = 1,
        n_beholder_keypoints: int = BEHOLDER_N_KEYPOINTS,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.n_keypoints = n_keypoints
        self.conf_threshold = confidence_threshold
        self.device = device
        self._features: torch.Tensor | None = None

        self.multiclass = multiclass
        if multiclass:
            self.max_partners = max_partners
            self.max_beholders = max_beholders
            self.max_persons = max_partners + max_beholders
            self.n_beholder_keypoints = n_beholder_keypoints
        else:
            self.max_persons = max_persons

        self._hook = model.model.model[layer_idx].register_forward_hook(self._capture)
        layer_name = type(model.model.model[layer_idx]).__name__
        log.info(
            "Registered hook on layer %d (%s)%s",
            layer_idx, layer_name,
            f" [multiclass: {max_partners}p+{max_beholders}b]" if multiclass else "",
        )

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

        In multiclass mode, slots 0..max_partners-1 hold partner detections
        and slots max_partners..max_persons-1 hold beholder detections.
        Beholder keypoints beyond index n_beholder_keypoints are zeroed.
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
            iou=0.97,
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

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            det_conf = boxes.conf.cpu().numpy()
            boxes_xyxy = boxes.xyxy  # [n_det, 4] stays on device for ROI align

            if self.multiclass:
                cls = boxes.cls.cpu().numpy().astype(int)  # [n_det]
                self._fill_multiclass(
                    i, kp_out, emb_out,
                    kpts_data, conf_data, det_conf, cls,
                    boxes_xyxy, features,
                )
            else:
                self._fill_single_class(
                    i, kp_out, emb_out,
                    kpts_data, conf_data, det_conf,
                    boxes_xyxy, features,
                )

        return kp_out, emb_out

    def _fill_single_class(
        self,
        frame_idx: int,
        kp_out: np.ndarray,
        emb_out: np.ndarray,
        kpts_data: np.ndarray,
        conf_data: np.ndarray,
        det_conf: np.ndarray,
        boxes_xyxy: torch.Tensor,
        features: torch.Tensor | None,
    ) -> None:
        """Fill keypoint/embedding arrays for single-class (original) mode."""
        sorted_idx = np.argsort(-det_conf)
        n_persons = min(len(kpts_data), self.max_persons)

        for j in range(n_persons):
            idx = sorted_idx[j]
            kp_out[frame_idx, j, :, :2] = kpts_data[idx]
            kp_out[frame_idx, j, :, 2] = conf_data[idx]

        if features is not None and len(boxes_xyxy) > 0:
            selected_boxes = boxes_xyxy[sorted_idx[:n_persons]]
            roi_features = roi_align(
                features[frame_idx : frame_idx + 1],
                [selected_boxes],
                output_size=ROI_OUTPUT_SIZE,
                spatial_scale=1.0 / BACKBONE_STRIDE,
                aligned=True,
            )
            embeddings = roi_features.mean(dim=[2, 3])
            emb_out[frame_idx, :n_persons] = embeddings.detach().cpu().numpy()

    def _fill_multiclass(
        self,
        frame_idx: int,
        kp_out: np.ndarray,
        emb_out: np.ndarray,
        kpts_data: np.ndarray,
        conf_data: np.ndarray,
        det_conf: np.ndarray,
        cls: np.ndarray,
        boxes_xyxy: torch.Tensor,
        features: torch.Tensor | None,
    ) -> None:
        """Fill keypoint/embedding arrays for multiclass mode.

        Partners (class 0) → slots 0..max_partners-1
        Beholders (class 1) → slots max_partners..max_persons-1
        """
        # Separate by class
        partner_mask = cls == PARTNER_CLASS
        beholder_mask = cls == BEHOLDER_CLASS

        partner_indices = np.where(partner_mask)[0]
        beholder_indices = np.where(beholder_mask)[0]

        # Sort each class by detection confidence (descending)
        if len(partner_indices) > 0:
            partner_order = partner_indices[np.argsort(-det_conf[partner_indices])]
        else:
            partner_order = np.array([], dtype=int)

        if len(beholder_indices) > 0:
            beholder_order = beholder_indices[np.argsort(-det_conf[beholder_indices])]
        else:
            beholder_order = np.array([], dtype=int)

        n_partners = min(len(partner_order), self.max_partners)
        n_beholders = min(len(beholder_order), self.max_beholders)

        # Fill partner keypoints (slots 0..n_partners-1, all 21 keypoints)
        for j in range(n_partners):
            idx = partner_order[j]
            kp_out[frame_idx, j, :, :2] = kpts_data[idx]
            kp_out[frame_idx, j, :, 2] = conf_data[idx]

        # Fill beholder keypoints (slot max_partners.., only first n_beholder_keypoints)
        for j in range(n_beholders):
            idx = beholder_order[j]
            slot = self.max_partners + j
            n_bkp = self.n_beholder_keypoints
            kp_out[frame_idx, slot, :n_bkp, :2] = kpts_data[idx, :n_bkp]
            kp_out[frame_idx, slot, :n_bkp, 2] = conf_data[idx, :n_bkp]
            # Indices n_bkp..20 stay zero (from initialization)

        # RoI Align embeddings from backbone features
        if features is not None and (n_partners > 0 or n_beholders > 0):
            roi_box_list = []
            roi_slot_map = []

            for j in range(n_partners):
                roi_box_list.append(boxes_xyxy[partner_order[j]])
                roi_slot_map.append(j)

            for j in range(n_beholders):
                roi_box_list.append(boxes_xyxy[beholder_order[j]])
                roi_slot_map.append(self.max_partners + j)

            if roi_box_list:
                selected_boxes = torch.stack(roi_box_list)
                roi_features = roi_align(
                    features[frame_idx : frame_idx + 1],
                    [selected_boxes],
                    output_size=ROI_OUTPUT_SIZE,
                    spatial_scale=1.0 / BACKBONE_STRIDE,
                    aligned=True,
                )
                emb_np = roi_features.mean(dim=[2, 3]).detach().cpu().numpy()
                for k, slot in enumerate(roi_slot_map):
                    emb_out[frame_idx, slot] = emb_np[k]

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
