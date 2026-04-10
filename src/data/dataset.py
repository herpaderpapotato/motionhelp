"""PyTorch Dataset for embeddings/pose + flow → funscript training."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class FunscriptDataset(Dataset):
    """Dataset that loads pre-extracted features and funscript labels.

    Each sample is a sequence of `seq_len` consecutive frames with configurable
    feature combinations:
    - Embeddings: [seq_len, max_persons * embed_dim] (YOLO model.embed features)
    - Pose keypoints: [seq_len, max_persons * n_keypoints * 3] (optional, legacy)
    - Optical flow features: [seq_len, flow_dim] (optional)
    - Labels: [seq_len] position values in [0, 1]
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = 120,
        stride: int = 60,
        use_pose: bool = False,
        use_embeddings: bool = True,
        use_flow: bool = True,
        max_persons: int = 2,
        embed_dim: int = 512,
        n_keypoints: int = 17,
        model_name: str = "yolo11m-pose",  # for new-layout path resolution
        flow_method: str = "raft",
        flow_features: int = 64,
        flow_scale: float = 0.5,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.use_pose = use_pose
        self.use_embeddings = use_embeddings
        self.use_flow = use_flow
        self.max_persons = max_persons
        self.embed_dim = embed_dim
        self.embedding_features = max_persons * embed_dim
        self.n_keypoints = n_keypoints
        self.model_name = model_name
        self.flow_method = flow_method
        self.flow_features = flow_features
        self.flow_scale = flow_scale
        self.augment = augment

        # Load split file
        split_file = self.data_dir / "splits" / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.video_ids = json.load(f)
        else:
            # Auto-discover from processed directory
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                self.video_ids = [d.name for d in processed_dir.iterdir() if d.is_dir()]
            else:
                self.video_ids = []

        # Load feature normalization stats (z-score)
        stats_path = self.data_dir / "feature_stats.npz"
        if stats_path.exists():
            stats = np.load(stats_path)
            self._emb_mean = stats["emb_mean"] if "emb_mean" in stats else None
            self._emb_std = stats["emb_std"] if "emb_std" in stats else None
            self._flow_mean = stats["flow_mean"] if "flow_mean" in stats else None
            self._flow_std = stats["flow_std"] if "flow_std" in stats else None

            if self._emb_mean is not None:
                if self._emb_mean.shape != (self.embedding_features,) or self._emb_std is None or self._emb_std.shape != (self.embedding_features,):
                    log.warning(
                        "Ignoring embedding stats from %s: expected shape (%d,), got mean=%s std=%s",
                        stats_path,
                        self.embedding_features,
                        None if self._emb_mean is None else self._emb_mean.shape,
                        None if self._emb_std is None else self._emb_std.shape,
                    )
                    self._emb_mean = self._emb_std = None

            if self._flow_mean is not None:
                if self._flow_mean.shape != (self.flow_features,) or self._flow_std is None or self._flow_std.shape != (self.flow_features,):
                    log.warning(
                        "Ignoring flow stats from %s: expected shape (%d,), got mean=%s std=%s",
                        stats_path,
                        self.flow_features,
                        None if self._flow_mean is None else self._flow_mean.shape,
                        None if self._flow_std is None else self._flow_std.shape,
                    )
                    self._flow_mean = self._flow_std = None
            if self._emb_mean is not None or self._flow_mean is not None:
                log.info("Loaded compatible feature normalization stats from %s", stats_path)
            else:
                log.warning("No compatible feature normalization stats found in %s — features will NOT be normalized", stats_path)
        else:
            self._emb_mean = self._emb_std = None
            self._flow_mean = self._flow_std = None
            log.warning("No feature_stats.npz found — features will NOT be normalized")
        # Load split file
        split_file = self.data_dir / "splits" / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.video_ids = json.load(f)
        else:
            # Auto-discover from processed directory
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                self.video_ids = [d.name for d in processed_dir.iterdir() if d.is_dir()]
            else:
                self.video_ids = []
        # Build index of all valid sequences
        self.sequences: list[tuple[str, int]] = []  # (video_id, start_frame)
        self._build_index()

    def _build_index(self):
        """Find all valid sequences across all videos."""
        from .curation import inspect_embeddings_path, resolve_keypoints_path

        for vid_id in self.video_ids:
            vid_dir = self.data_dir / "processed" / vid_id
            labels_path = vid_dir / "labels.npy"

            if not labels_path.exists():
                log.warning("No labels found for %s, skipping", vid_id)
                continue

            # Check required features exist (new paths + legacy fallback)
            if self.use_embeddings:
                emb_path, emb_reason = inspect_embeddings_path(
                    vid_dir,
                    self.model_name,
                    max_persons=self.max_persons,
                    require_current=True,
                )
                if emb_path is None:
                    log.warning("Embeddings unavailable for %s (%s), skipping", vid_id, emb_reason)
                    continue

            if self.use_embeddings and emb_path is None:
                continue

            if self.use_pose:
                pose_path = resolve_keypoints_path(vid_dir, self.model_name)
                if pose_path is None:
                    log.warning("No pose keypoints found for %s, skipping", vid_id)
                    continue

                pose_data = np.load(str(pose_path), mmap_mode="r")
                if pose_data.shape[2] != self.n_keypoints:
                    log.warning(
                        "Pose keypoints in %s have %d keypoints, expected %d. Skipping video.",
                        vid_id, pose_data.shape[2], self.n_keypoints)
                    continue

            labels = np.load(labels_path)
            n_frames = len(labels)

            for start in range(0, n_frames - self.seq_len + 1, self.stride):
                self.sequences.append((vid_id, start))

        log.info("Built dataset index: %d sequences from %d videos",
                 len(self.sequences), len(self.video_ids))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        vid_dir = self.data_dir / "processed" / vid_id
        end = start + self.seq_len

        # Load labels: [N]
        labels = np.load(vid_dir / "labels.npy", mmap_mode="r")[start:end].copy()

        sample = {
            "labels": torch.from_numpy(labels).float(),     # [seq_len]
        }

        # Load YOLO embeddings: [N, max_persons, embed_dim]
        if self.use_embeddings:
            from .curation import inspect_embeddings_path
            emb_path, emb_reason = inspect_embeddings_path(
                vid_dir,
                self.model_name,
                max_persons=self.max_persons,
                require_current=True,
            )
            if emb_path is None:
                raise RuntimeError(
                    f"Embeddings missing or stale for {vid_id}: {emb_reason}"
                )
            emb_data = np.load(str(emb_path), mmap_mode="r")[start:end]  # [seq_len, max_persons, embed_dim]
            emb_flat = emb_data[:, :self.max_persons].reshape(self.seq_len, -1).copy()
            # Z-score normalize embeddings if stats are available
            if self._emb_mean is not None and self._emb_mean.shape[0] == emb_flat.shape[1]:
                emb_flat = (emb_flat - self._emb_mean) / (self._emb_std + 1e-8)
            sample["embeddings"] = torch.from_numpy(emb_flat).float()

        # Load pose keypoints: [N, max_persons, n_keypoints, 3] (optional)
        if self.use_pose:
            from .curation import resolve_keypoints_path
            pose_path = resolve_keypoints_path(vid_dir, self.model_name)
            if pose_path is None:
                raise RuntimeError(
                    "Pose keypoints missing for %s but video was indexed as valid" % vid_id)

            pose_data = np.load(str(pose_path), mmap_mode="r")[start:end]
            if pose_data.shape[2] != self.n_keypoints:
                raise RuntimeError(
                    "Pose keypoints in %s have %d keypoints, expected %d." % (
                        vid_id, pose_data.shape[2], self.n_keypoints))

            pose_flat = pose_data[:, :self.max_persons].reshape(self.seq_len, -1).copy()
            sample["pose"] = torch.from_numpy(pose_flat).float()

        # Load optical flow if available
        if self.use_flow:
            from .curation import resolve_flow_path
            flow_path = resolve_flow_path(
                vid_dir, self.flow_method, self.flow_features, self.flow_scale)
            if flow_path is not None:
                flow_data = np.load(str(flow_path), mmap_mode="r")[start:end].copy()
                # Z-score normalize flow if stats are available
                if self._flow_mean is not None and self._flow_mean.shape[0] == flow_data.shape[1]:
                    flow_data = (flow_data - self._flow_mean) / (self._flow_std + 1e-8)
                sample["flow"] = torch.from_numpy(flow_data).float()
            else:
                sample["flow"] = torch.zeros(self.seq_len, self.flow_features)

        # Augmentation
        if self.augment:
            sample = self._augment(sample)

        return sample

    def _augment(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply data augmentation to a sample."""
        all_feature_keys = [k for k in ("pose", "embeddings", "flow") if k in sample]

        # Time reversal (50% chance)
        if torch.rand(1).item() < 0.5:
            for key in all_feature_keys + ["labels"]:
                sample[key] = sample[key].flip(0)
            if "flow" in sample:
                sample["flow"] = -sample["flow"]  # Reverse flow direction

        # Position inversion (50% chance) — flip 0↔1
        if torch.rand(1).item() < 0.5:
            sample["labels"] = 1.0 - sample["labels"]
            # Flip pose Y coordinates if pose present
            if "pose" in sample:
                pose = sample["pose"].view(self.seq_len, -1, 3)  # [T, n_kpts, 3]
                pose[:, :, 1] = 1.0 - pose[:, :, 1]  # Flip Y
                sample["pose"] = pose.view(self.seq_len, -1)

        # Embedding feature dropout (25% chance) — zero out random feature dims
        if "embeddings" in sample and torch.rand(1).item() < 0.25:
            emb = sample["embeddings"]
            # Drop random dimensions across all frames
            dim_mask = torch.rand(emb.shape[-1]) < 0.1
            emb[:, dim_mask] = 0.0
            sample["embeddings"] = emb

        # Embedding noise (30% chance) — small gaussian perturbation
        if "embeddings" in sample and torch.rand(1).item() < 0.3:
            emb = sample["embeddings"]
            noise_scale = emb.std() * 0.02  # 2% of feature std
            noise = torch.randn_like(emb) * noise_scale
            sample["embeddings"] = emb + noise

        # Pose keypoint noise (40% chance) — if pose features present
        if "pose" in sample and torch.rand(1).item() < 0.4:
            pose = sample["pose"].view(self.seq_len, -1, 3)
            coord_noise = torch.randn(self.seq_len, pose.shape[1], 2) * 0.01
            pose[:, :, :2] = torch.clamp(pose[:, :, :2] + coord_noise, 0.0, 1.0)
            sample["pose"] = pose.view(self.seq_len, -1)

        # Confidence dropout (20% chance) — if pose features present
        if "pose" in sample and torch.rand(1).item() < 0.2:
            pose = sample["pose"].view(self.seq_len, -1, 3)
            drop_mask = torch.rand(self.seq_len, pose.shape[1]) < 0.1
            pose[drop_mask] = 0.0
            sample["pose"] = pose.view(self.seq_len, -1)

        # Small temporal noise on labels (30% chance)
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(sample["labels"]) * 0.02
            sample["labels"] = torch.clamp(sample["labels"] + noise, 0.0, 1.0)

        # Speed perturbation (15% chance) — slight stretch/compress in time
        if torch.rand(1).item() < 0.15:
            scale = 0.9 + torch.rand(1).item() * 0.2  # 0.9-1.1x
            indices = torch.linspace(0, self.seq_len - 1, int(self.seq_len * scale)).long()
            indices = torch.clamp(indices, 0, self.seq_len - 1)
            # Resample to original length
            for key in all_feature_keys + ["labels"]:
                resampled = sample[key][indices]
                if len(resampled) >= self.seq_len:
                    sample[key] = resampled[:self.seq_len]
                else:
                    pad = sample[key][-1:].expand(self.seq_len - len(resampled), *sample[key].shape[1:])
                    sample[key] = torch.cat([resampled, pad], dim=0)

        return sample


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate function for FunscriptDataset."""
    result = {}
    for key in batch[0]:
        result[key] = torch.stack([sample[key] for sample in batch])
    return result
