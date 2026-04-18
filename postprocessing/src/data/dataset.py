"""Dataset for postprocessing refinement training.

Loads prediction/label pairs prepared by prepare_data.py.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class RefinementDataset(Dataset):
    """Dataset of (prediction, label) pairs for refinement training.

    Each sample is a contiguous subsequence of `seq_len` frames from a
    prepared scene.  When `seq_len` is None the full sequence is returned.
    """

    def __init__(
        self,
        prepared_dir: str | Path,
        splits_dir: str | Path,
        split: str = "train",
        seq_len: int | None = None,
        stride: int | None = None,
        augment: bool = False,
    ):
        self.prepared_dir = Path(prepared_dir)
        self.seq_len = seq_len
        self.stride = stride or (seq_len // 2 if seq_len else None)
        self.augment = augment

        # Load split
        split_file = Path(splits_dir) / f"{split}.json"
        with open(split_file) as f:
            scene_ids = json.load(f)

        # Build index
        self.sequences: list[tuple[str, int, int]] = []  # (scene_id, start, length)

        for scene_id in scene_ids:
            pred_path = self.prepared_dir / scene_id / "predictions.npy"
            label_path = self.prepared_dir / scene_id / "labels.npy"
            if not pred_path.exists() or not label_path.exists():
                continue

            n_frames = len(np.load(str(pred_path), mmap_mode="r"))

            if seq_len is None:
                self.sequences.append((scene_id, 0, n_frames))
            else:
                for start in range(0, n_frames - seq_len + 1, self.stride):
                    self.sequences.append((scene_id, start, seq_len))

        log.info("RefinementDataset[%s]: %d sequences from %d scenes",
                 split, len(self.sequences), len(scene_ids))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        scene_id, start, length = self.sequences[idx]
        end = start + length

        predictions = np.load(
            str(self.prepared_dir / scene_id / "predictions.npy"), mmap_mode="r"
        )[start:end].copy()
        labels = np.load(
            str(self.prepared_dir / scene_id / "labels.npy"), mmap_mode="r"
        )[start:end].copy()

        if self.augment:
            # Random flip: invert both predictions and labels
            if np.random.random() < 0.5:
                predictions = 1.0 - predictions
                labels = 1.0 - labels

        return {
            "predictions": torch.from_numpy(predictions).float(),
            "labels": torch.from_numpy(labels).float(),
        }
