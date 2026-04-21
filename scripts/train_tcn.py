"""Training script for the TCN funscript prediction model.

Usage:
    python scripts/train_tcn.py --epochs 100
    python scripts/train_tcn.py --epochs 5 --batch-size 32  # quick test

Loads pre-extracted features (keypoints, embeddings, flow) from data/processed/
and trains a temporal convolutional network for per-frame position prediction.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Resolve imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.embeddings import load_embeddings_metadata
from src.models.tcn import FunscriptTCN, extract_model_config
from src.training.funscript_metrics import compute_regression_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

log = logging.getLogger(__name__)

DEFAULT_MULTICLASS_FEATURE_MODEL = "vrlens-finetunes-multiclass-v2-yolo11m-pose"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MotionDataset(Dataset):
    """Loads keypoints, embeddings, flow, and labels for TCN training.

    Expects data layout:
        data/processed/{scene_id}/labels.npy                              [T]
        data/processed/{scene_id}/keypoints/{model_name}.npy              [T, N, K, 3]
        data/processed/{scene_id}/embeddings/{model_name}.npy             [T, N, E]
        data/processed/{scene_id}/flow/raft_f64_s0.5.npy                  [T, F]
    """

    FLOW_FILE = "flow/raft_f64_s0.5.npy"
    DENSE_FLOW_FILE = "flow/raft_dense_32x32_s0.5.npy"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        seq_len: int = 120,
        stride: int = 60,
        n_persons: int = 10,
        n_keypoints: int = 21,
        embed_dim: int | None = None,
        flow_dim: int = 64,
        augment: bool = False,
        multiclass: bool = False,
        feature_model_name: str = DEFAULT_MULTICLASS_FEATURE_MODEL,
        phase: int = -1,
        flow_mode: str = "summary",
        flow_dense_size: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.n_persons = n_persons
        self.n_keypoints = n_keypoints
        self.flow_dim = flow_dim
        self.augment = augment
        self.multiclass = multiclass
        self.feature_model_name = Path(feature_model_name).stem
        self.phase = phase
        self.flow_mode = flow_mode
        self.flow_dense_size = flow_dense_size
        self.using_stats = False
        self.stats_path = None
        self.last_epoch = False

        if multiclass:
            self.KP_FILE = str(Path("keypoints") / f"{self.feature_model_name}.npy")
            self.EMB_FILE = str(Path("embeddings") / f"{self.feature_model_name}.npy")
            self.EMB_FILE_META = str(Path("embeddings") / f"{self.feature_model_name}.json")
        else:
            self.KP_FILE = "keypoints/pose-vrlens-finetunes-large.npy"
            self.EMB_FILE = "embeddings/pose-vrlens-finetunes-large.npy"
            self.EMB_FILE_META = "embeddings/pose-vrlens-finetunes-large.json"
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.n_persons = n_persons
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim
        self.augment = augment
        self.augment_scale: float = 0.0  # 0=min augmentation, 1=max augmentation

        # Load split
        with open(self.data_dir / "splits" / f"{split}.json") as f:
            video_ids = json.load(f)

        if embed_dim is None:
            self.embed_dim = self._infer_embed_dim(video_ids) if multiclass else 512
        else:
            self.embed_dim = embed_dim

        # Load normalization stats
        self._load_stats()

        # Build sequence index (only complete scenes)
        self.sequences: list[tuple[str, int]] = []
        processed = self.data_dir / "processed"
        skipped = 0

        for vid_id in video_ids:
            vid_dir = processed / vid_id
            kp_path = vid_dir / self.KP_FILE
            emb_path = vid_dir / self.EMB_FILE
            emb_meta_path = vid_dir / self.EMB_FILE_META
            flow_path = vid_dir / self.FLOW_FILE
            label_path = vid_dir / "labels.npy"

            if self.flow_mode == "dense":
                dense_flow_p = vid_dir / self.DENSE_FLOW_FILE
                required = [kp_path, emb_path, emb_meta_path, dense_flow_p, label_path]
            else:
                required = [kp_path, emb_path, emb_meta_path, flow_path, label_path]
            if not all(p.exists() for p in required):
                skipped += 1
                continue
            
            # check for review.json in scene directory and skip if "status": "rejected" or "stage2_status": "rejected"
            review_path = vid_dir / "review.json"
            if review_path.exists():
                review = json.loads(review_path.read_text(encoding="utf-8"))
                if review.get("status") == "rejected" or review.get("stage2_status") == "rejected":
                    skipped += 1
                    continue
                # if float(review.get("mse", 0)) > 0.02 and self.augment:
                #     skipped += 1
                #     continue
            # check embedding metadata is valid for current extraction method
            if emb_meta_path.exists():
                meta = json.loads(emb_meta_path.read_text(encoding="utf-8"))
                if meta.get("method") != "single_pass_hook_roi_align":
                    skipped += 1
                    continue
                if self.multiclass and not meta.get("multiclass", False):
                    skipped += 1
                    continue
                if int(meta.get("embed_dim", self.embed_dim)) != self.embed_dim:
                    skipped += 1
                    continue
            


            n_frames = np.load(str(label_path), mmap_mode="r").shape[0]
            if n_frames < seq_len:
                skipped += 1
                continue

            for start in range(0, n_frames - seq_len + 1, stride):
                self.sequences.append((vid_id, start))

        log.info(
            "Built %s dataset: %d sequences from %d videos (%d skipped)",
            split, len(self.sequences), len(video_ids) - skipped, skipped,
        )

    def sample_high_variance(self, n: int = 8, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Return n sequences with the highest label variance from a random pool."""
        pool = min(2000, len(self.sequences))
        idx = np.random.choice(len(self.sequences), pool, replace=False)
        stds = []
        for i in idx:
            vid_id, start = self.sequences[i]
            lbl = np.load(str(self.data_dir / "processed" / vid_id / "labels.npy"),
                          mmap_mode="r")[start:start + self.seq_len]
            stds.append(float(lbl.std()))
        top_idx = np.array(stds).argsort()[-n:][::-1]
        batch = [self[idx[i]] for i in top_idx]
        out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out
    
    def sample_random(self, n: int = 8, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Return a random batch of n sequences."""
        idx = np.random.choice(len(self.sequences), n, replace=False)
        batch = [self[i] for i in idx]
        out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out




    def _infer_embed_dim(self, video_ids: list[str]) -> int:
        processed = self.data_dir / "processed"
        for vid_id in video_ids:
            emb_path = processed / vid_id / self.EMB_FILE
            metadata = load_embeddings_metadata(emb_path)
            if metadata is not None and metadata.get("embed_dim") is not None:
                try:
                    return int(metadata["embed_dim"])
                except (TypeError, ValueError):
                    pass
            if emb_path.exists():
                try:
                    emb_shape = np.load(str(emb_path), mmap_mode="r").shape
                except Exception:
                    continue
                if len(emb_shape) == 3:
                    return int(emb_shape[2])

        log.warning(
            "Could not infer embedding width for multiclass feature model %s; defaulting to 512",
            self.feature_model_name,
        )
        return 512

    def _load_stats(self) -> None:
        stats_candidates: list[Path] = []
        if self.multiclass:
            feature_model_name = Path(self.feature_model_name).stem
            stats_candidates.extend(
                [
                    self.data_dir / f"feature_stats_{feature_model_name}.npz",
                    self.data_dir / "featurestats" / f"feature_stats_{feature_model_name}.npz",
                ]
            )
        stats_candidates.extend(
            [
                self.data_dir / "featurestats" / "feature_stats.npz",
                self.data_dir / "feature_stats.npz",
            ]
        )
        stats_path = next((candidate for candidate in stats_candidates if candidate.exists()), None)
        self.using_stats = False
        if stats_path is None:
            log.warning("No feature stats file found — features will NOT be normalized")
            self.emb_mean = self.emb_std = None
            self.flow_mean = self.flow_std = None
            return
        self.stats_path = stats_path
        self.using_stats = True

        stats = np.load(stats_path)
        emb_mean = stats.get("emb_mean")
        emb_std = stats.get("emb_std")

        # Reshape [N*E] → [N, E] for per-person normalization
        expected = self.n_persons * self.embed_dim
        if emb_mean is not None and emb_mean.shape[0] == expected:
            self.emb_mean = emb_mean.reshape(self.n_persons, self.embed_dim)
            self.emb_std = emb_std.reshape(self.n_persons, self.embed_dim)
            log.info(
                "Loaded embedding normalization stats: [%d persons x %d dims] from %s",
                self.n_persons, self.embed_dim, stats_path,
            )
        elif emb_mean is not None and emb_mean.shape[0] == self.embed_dim:
            self.emb_mean = emb_mean
            self.emb_std = emb_std
            log.info("Loaded embedding normalization stats: [%d dims] from %s", self.embed_dim, stats_path)
        else:
            self.emb_mean = self.emb_std = None
            
            self.using_stats = False
            log.warning(
                "Embedding stats shape %s does not match expected (%d,) or (%d,) — not normalizing embeddings",
                emb_mean.shape if emb_mean is not None else "None",
                expected, self.embed_dim,
            )

        flow_mean = stats.get("flow_mean")
        flow_std = stats.get("flow_std")
        if flow_mean is not None and flow_mean.shape[0] == self.flow_dim:
            self.flow_mean = flow_mean
            self.flow_std = flow_std
            log.info("Loaded flow normalization stats: [%d dims] from %s", self.flow_dim, stats_path)
        else:
            self.flow_mean = self.flow_std = None
            
            self.using_stats = False
            log.warning("Flow stats not found or shape mismatch — not normalizing flow")

        # Dense flow stats
        self.flow_dense_mean = None
        self.flow_dense_std = None
        if self.flow_mode == "dense":
            flow_dense_mean = stats.get("flow_dense_mean")
            flow_dense_std = stats.get("flow_dense_std")
            if flow_dense_mean is not None:
                self.flow_dense_mean = flow_dense_mean  # [2, 1, 1]
                self.flow_dense_std = flow_dense_std
                log.info("Loaded dense flow normalization stats: shape=%s from %s",
                         flow_dense_mean.shape, stats_path)
            else:
                log.warning("Dense flow stats not found — dense flow will NOT be normalized")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        vid_dir = self.data_dir / "processed" / vid_id
        end = start + self.seq_len

        labels = np.load(str(vid_dir / "labels.npy"), mmap_mode="r")[start:end].copy()
        keypoints = np.load(str(vid_dir / self.KP_FILE), mmap_mode="r")[start:end].copy()
        embeddings = np.load(str(vid_dir / self.EMB_FILE), mmap_mode="r")[start:end].copy()

        # Normalize embeddings
        if self.emb_mean is not None:
            embeddings = (embeddings - self.emb_mean) / (self.emb_std + 1e-8)

        if self.flow_mode == "dense":
            flow = np.load(str(vid_dir / self.DENSE_FLOW_FILE), mmap_mode="r")[start:end].copy().astype(np.float32)
            # Normalize dense flow: [T, 2, H, W] with stats [2, 1, 1]
            if self.flow_dense_mean is not None:
                flow = (flow - self.flow_dense_mean) / (self.flow_dense_std + 1e-8)
        else:
            flow = np.load(str(vid_dir / self.FLOW_FILE), mmap_mode="r")[start:end].copy()
            if self.flow_mean is not None:
                flow = (flow - self.flow_mean) / (self.flow_std + 1e-8)

        keypoints = torch.from_numpy(keypoints).float()    # [T, N, K, 3]
        embeddings = torch.from_numpy(embeddings).float()   # [T, N, E]
        flow = torch.from_numpy(flow).float()               # [T, F]
        labels = torch.from_numpy(labels).float()            # [T]

        if self.augment or self.phase > 1:
            keypoints, embeddings, flow, labels = self._augment(
                keypoints, embeddings, flow, labels
            )

        return {
            "keypoints": keypoints,
            "embeddings": embeddings,
            "flow": flow,
            "labels": labels,
        }

    def _augment(
        self,
        kp: torch.Tensor,
        emb: torch.Tensor,
        flow: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # All augmentations are independent (not elif) so multiple can apply

        if not self.last_epoch:
            if self.phase == 2:
                # null out all inputs except pose keypoints to focus on learning from pose
                emb = torch.zeros_like(emb)
                flow = torch.zeros_like(flow)
            elif self.phase == 3:
                # null out all inputs except embeddings to focus on learning from embeddings
                kp = torch.zeros_like(kp)
                flow = torch.zeros_like(flow)
            elif self.phase == 4:
                # null out all inputs except flow to focus on learning from flow
                kp = torch.zeros_like(kp)
                emb = torch.zeros_like(emb)

        s = self.augment_scale  # [0, 1] — 0=min values, 1=max values

        dropout_prob = 0.1 + s * (0.9 - 0.1)
        if torch.rand(1).item() < dropout_prob:
            choice = torch.randint(3, (1,)).item()
            if choice == 0:
                kp = torch.zeros_like(kp)
            elif choice == 1:
                emb = torch.zeros_like(emb)
            else:
                flow = torch.zeros_like(flow)

        emb_noise_prob = 0.2 + s * (0.9 - 0.2)    
        emb_noise_mag  = 0.02 + s * (0.1 - 0.02)
        if torch.rand(1).item() < emb_noise_prob:
            emb = emb + torch.randn_like(emb) * emb_noise_mag

        # flow_noise_prob = 0.2 + s * (0.5 - 0.2) 
        # flow_noise_mag  = 0.02 + s * (0.1- 0.02)
        # if torch.rand(1).item() < flow_noise_prob:
        #     # flow = flow + torch.randn_like(flow) * flow_noise_mag # makes B x Seq x FlowDim noise which may be too random
        #     # Alternate: use same noise for all flow frames in a sequence to generalize better to different flow magnitudes
        #     noise = torch.randn(flow.shape[1], device=flow.device) * flow_noise_mag
        #     flow = flow + noise



        return kp, emb, flow, labels


def _resolve_sequence_limit(total_sequences: int, requested_limit: int | None) -> int:
    if requested_limit is None:
        return total_sequences
    return min(total_sequences, requested_limit)


class EpochSubsetSampler(Sampler[int]):
    # Re-samples a capped subset each epoch when shuffle=True.
    def __init__(
        self,
        dataset_size: int,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset_size = dataset_size
        self.max_samples = _resolve_sequence_limit(dataset_size, max_samples)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.max_samples

    def __iter__(self):
        if not self.shuffle:
            return iter(range(self.max_samples))

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator)[:self.max_samples]
        return iter(indices.tolist())


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _compute_augment_scale(val_loss: float, train_loss: float) -> float:
    """Map val/train loss ratio to augmentation scale [0, 1].

    Linear interpolation in between.
    """
    start_threshold = 1.0
    end_threshold = 2.0
    if train_loss <= 0:
        return 0.0
    ratio = val_loss / train_loss
    if ratio < start_threshold:
        return 0.0
    if ratio >= end_threshold:
        return 1.0
    return (ratio - start_threshold) / (end_threshold - start_threshold)


def _compute_loss_from_multichannel(
    pred: torch.Tensor,
    lbl: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute combined loss from [B, 4, T] model output.

    Returns (total_loss, main_pred [B, T], metric_dict).
    Channel 0 = fused, 1 = pose-only, 2 = emb-only, 3 = flow-only.
    """
    main_pred = pred[:, 0]  # [B, T] — fused prediction

    metric_batch = compute_regression_metrics(
        main_pred,
        lbl,
        spectral_kernel=args.spectral_kernel,
        activity_gain=args.event_activity_gain,
        activity_power=args.event_activity_power,
        active_quantile=args.active_quantile,
    )
    pos_loss = ((1.0 - args.event_weight) * metric_batch["pos_mse"]
                + args.event_weight * metric_batch["event_mse"])
    temp_loss = metric_batch["acc_mse"]
    vel_loss = metric_batch["vel_mse"]
    spec_loss = metric_batch["spec_mse"]

    loss = (pos_loss
            + args.temporal_weight * temp_loss
            + args.velocity_weight * vel_loss
            + args.spectral_weight * spec_loss)
    
    if args.use_aux_layers:
        # Auxiliary branch losses (MSE against same labels)
        if args.aux_weight > 0:
            aux_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
            for ch in range(1, 4):
                aux_pred = pred[:, ch]  # [B, T]
                aux_loss = aux_loss + F.mse_loss(aux_pred, lbl)
            aux_loss = aux_loss / 3.0  # average across 3 auxiliary branches
            loss = loss + args.aux_weight * aux_loss
            metric_batch["aux_loss"] = aux_loss.detach()

    return loss, main_pred, metric_batch


# ---------------------------------------------------------------------------
# Phase-specific freezing for multi-phase training
# ---------------------------------------------------------------------------

def _apply_phase_freezing(model: FunscriptTCN, phase: int, fill_with_noise: bool = False) -> None:
    """Freeze/unfreeze model parameters based on training phase.
    """
    # First, freeze everything
    if phase > 1:
        for p in model.parameters():
            p.requires_grad = False
    if phase == 1:
        pass  # all trainable
    elif phase == 2:
        if fill_with_noise:
            print("Filling pose encoder with noise for phase 2")
            for p in model.pose_encoder.parameters():
                p.data = torch.randn_like(p.data) * 0.01
            for p in model.pose_attn.parameters():
                p.data = torch.randn_like(p.data) * 0.01
            if model.multiclass:
                for p in model.beholder_pose_encoder.parameters():
                    p.data = torch.randn_like(p.data) * 0.01
        # Unfreeze pose encoders only
        print("Unfreezing pose encoders only")
        for p in model.pose_encoder.parameters():
            p.requires_grad = True
        for p in model.pose_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_pose_encoder.parameters():
                p.requires_grad = True
    elif phase == 3:
        if fill_with_noise:
            print("Filling embedding encoders with noise for phase 3")
            for p in model.emb_encoder.parameters():
                p.data = torch.randn_like(p.data) * 0.01
            for p in model.emb_attn.parameters():
                p.data = torch.randn_like(p.data) * 0.01
            if model.multiclass:
                for p in model.beholder_emb_encoder.parameters():
                    p.data = torch.randn_like(p.data) * 0.01
        # Unfreeze embedding encoders only
        print("Unfreezing embedding encoders only")
        for p in model.emb_encoder.parameters():
            p.requires_grad = True
        for p in model.emb_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_emb_encoder.parameters():
                p.requires_grad = True
    elif phase == 4:
        if fill_with_noise:
            print("Filling flow encoder with noise for phase 4")
            for p in model.flow_encoder.parameters():
                p.data = torch.randn_like(p.data) * 0.01
        # Unfreeze flow encoder only
        print("Unfreezing flow encoder only")
        for p in model.flow_encoder.parameters():
            p.requires_grad = True




    elif phase == 5:
        # Unfreeze flow encoder, fusion + output head + TCN backbone
        print("Unfreezing flow encoder, fusion + output head + TCN backbone")
        for p in model.flow_encoder.parameters():
            p.requires_grad = True 
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 6:
        # Unfreeze embedding encoders
        print("Unfreezing embedding encoders only")
        for p in model.emb_encoder.parameters():
            p.requires_grad = True
        for p in model.emb_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_emb_encoder.parameters():
                p.requires_grad = True
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 7:
        # Unfreeze pose encoders
        print("Unfreezing pose encoders only")
        for p in model.pose_encoder.parameters():
            p.requires_grad = True
        for p in model.pose_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_pose_encoder.parameters():
                p.requires_grad = True
        for p in model.flow_encoder.parameters():
            p.requires_grad = True 
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 8:
        # Unfreeze flow encoder, embedding encoders, fusion + output head + TCN backbone
        print("Unfreezing flow encoder, embedding encoders, fusion + output head + TCN backbone")
        for p in model.emb_encoder.parameters():
            p.requires_grad = True
        for p in model.emb_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_emb_encoder.parameters():
                p.requires_grad = True
        for p in model.flow_encoder.parameters():
            p.requires_grad = True 
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 9:
        if fill_with_noise:
            print("Filling fusion + output head + TCN backbone with noise for phase 9")
            for p in model.fusion.parameters():
                p.data = torch.randn_like(p.data) * 0.01
            for p in model.output_head.parameters():
                p.data = torch.randn_like(p.data) * 0.01
            for p in model.tcn_blocks.parameters():
                p.data = torch.randn_like(p.data) * 0.01

        # Unfreeze fusion + output head + TCN backbone
        print("Unfreezing fusion + output head + TCN backbone")
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 10:
        # Unfreeze pose encoders only
        print("Unfreezing pose encoders only")
        for p in model.pose_encoder.parameters():
            p.requires_grad = True
        for p in model.pose_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_pose_encoder.parameters():
                p.requires_grad = True
    elif phase == 11:
        # Unfreeze fusion + output head + TCN backbone
        print("Unfreezing fusion + output head + TCN backbone")
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 12:
        print("Phase 12: all trainable (but do it at a low lr)")
        pass  # all trainable
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1-12.")

    log.info("Phase %d freezing applied", phase)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    # wishlist
    # - when a job starts, dump the config for it to disk in root and in the job's checkpoint folder
    # - allow a resume last option
    # - use separate checkpoint folders with epoch time by default
    # - each epoch, read from an override config file that may override certain parameters like early stopping patience or augmentation scale or whatever, so that we can have a single job that runs through multiple phases with different configs without needing to manually intervene to change the config or launch new jobs
    # - a gui that shows the config of the current job and allows overriding certain parameters on the fly
    #  - also show current epoch # and mse, last best epoch # and mse, and previous best epochs
    #  - when changing config parameters on the fly, write the config to the config file with the epoch number in the json. e.g. "changes": [{"epoch": 5, "params": {"early_stopping_patience": 5, "augment_scale": 0.5}}] etc
    


    import random
    parser = argparse.ArgumentParser(description="Train TCN funscript model")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--max-train-sequences-per-epoch", type=int, default=None,
                        help="Cap the number of train sequences used in each epoch")
    parser.add_argument("--shuffle", action="store_true", dest="shuffle",
                        help="Shuffle train sequences and resample capped subsets each epoch")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle",
                        help="Disable shuffling and reuse the same deterministic subset each epoch")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="CosineWarmupLR",
                        choices=["OneCycleLR", "CosineAnnealingLR", "CosineWarmupLR", "ReduceLROnPlateau"])
    parser.add_argument("--multiclass", action="store_true",
                        help="Use multiclass model (partner + beholder)")
    parser.add_argument(
        "--feature-model-name",
        type=str,
        default=DEFAULT_MULTICLASS_FEATURE_MODEL,
        help="Multiclass feature model stem used for keypoint and embedding filenames",
    )
    parser.add_argument("--n-partners", type=int, default=5)
    parser.add_argument("--n-beholders", type=int, default=1)
    parser.add_argument("--n-beholder-keypoints", type=int, default=7)
    parser.add_argument("--use-kinematics", action="store_true",
                        help="Add velocity/acceleration features from keypoints")
    parser.add_argument("--use-ddl", action="store_true",
                        help="Use Dual Dilated Layers (MS-TCN++ style)")
    parser.add_argument("--kin-dim", type=int, default=64,
                        help="Kinematic feature dimension")
    parser.add_argument("--use-gated-fusion", action="store_true",
                        help="Use context-aware gated multimodal fusion instead of concat+linear")
    parser.add_argument("--no-difference-pathway", action="store_true", default=False,
                        help="Disable the multiclass beholder-performer difference branch")
    parser.add_argument("--difference-dim", type=int, default=64,
                        help="Feature dimension for the beholder-performer difference branch")
    parser.add_argument("--velocity-weight", type=float, default=0.9,
                        help="Weight for velocity-matching loss (first derivative)")
    parser.add_argument("--spectral-weight", type=float, default=0.0,
                        help="Weight for high-frequency detail loss (multi-scale spectral)")
    parser.add_argument("--spectral-kernel", type=int, default=15,
                        help="Moving-average kernel size for spectral loss low-pass filter")
    parser.add_argument("--event-weight", type=float, default=0.0,
                        help="Blend factor between plain MSE and event-aware weighted MSE")
    parser.add_argument("--event-activity-gain", type=float, default=3.0,
                        help="Gain applied to derivative-based event weighting")
    parser.add_argument("--event-activity-power", type=float, default=1.0,
                        help="Power applied to derivative-based event weighting")
    parser.add_argument("--active-quantile", type=float, default=0.8,
                        help="Quantile threshold used when reporting active-frame metrics")
    parser.add_argument("--phase", type=int, default=None,
                        help="Training phase (1-6) for multi-phase training")
    parser.add_argument("--fill-with-noise", action="store_true", default=False,
                        help="When resuming to a new phase, fill the newly unfrozen parts with noise instead of starting from the previous phase's weights")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint (for multi-phase training)")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop training if val loss has not improved for this many epochs (0 = disabled)")
    parser.add_argument("--load-best-val-loss", action="store_true", default=False,
                        help="When resuming, load the best_val_loss from the checkpoint to continue early stopping correctly")
    parser.add_argument("--flow-mode", type=str, default="summary",
                        choices=["summary", "dense"],
                        help="Flow representation: 'summary' (flat 64-d) or 'dense' (2×32×32 spatial)")
    parser.add_argument("--flow-dense-size", type=int, default=32,
                        help="Spatial resolution for dense flow maps (default: 32)")
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="Weight for auxiliary per-modality branch losses (0 = ignore aux branches)")
    parser.add_argument("--use-aux-layers", action="store_true", default=False,
                        help="Enable auxiliary per-modality branches")
    parser.add_argument("--disable-aux-layers", action="store_false", dest="use_aux_layers",
                        help="Disable auxiliary per-modality branches (overrides --use-aux-layers)")
    args = parser.parse_args()


    if not args.use_aux_layers:
        args.aux_weight = 0.0

    if args.max_train_sequences_per_epoch is not None and args.max_train_sequences_per_epoch <= 0:
        parser.error("--max-train-sequences-per-epoch must be a positive integer")
    if not 0.0 <= args.event_weight <= 1.0:
        parser.error("--event-weight must be between 0 and 1")
    if not 0.0 < args.active_quantile <= 1.0:
        parser.error("--active-quantile must be in (0, 1]")
    if args.early_stopping_patience < 5:
        args.early_stopping_patience = max(args.early_stopping_patience, 5)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    # Reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        print("Setting CUDA seeds for reproducibility")
        torch.cuda.manual_seed_all(args.seed)

    # ── Datasets ──────────────────────────────────────────────────────────
    n_total = (args.n_partners + args.n_beholders) if args.multiclass else 10

    train_ds = MotionDataset(
        args.data_dir, "train", args.seq_len, args.stride,
        n_persons=n_total,
        embed_dim=None if args.multiclass else 512,
        augment=True,
        multiclass=args.multiclass,
        feature_model_name=args.feature_model_name,
        phase=args.phase,
        flow_mode=args.flow_mode, flow_dense_size=args.flow_dense_size,
    )
    val_ds = MotionDataset(
        args.data_dir, "val", args.seq_len, 60,
        n_persons=n_total,
        embed_dim=train_ds.embed_dim,
        augment=False,
        multiclass=args.multiclass,
        feature_model_name=args.feature_model_name,
        flow_mode=args.flow_mode, flow_dense_size=args.flow_dense_size,
    )

    train_sequences_per_epoch = _resolve_sequence_limit(
        len(train_ds),
        args.max_train_sequences_per_epoch,
    )
    train_subset_is_limited = train_sequences_per_epoch < len(train_ds)
    val_sequences_per_epoch: int | None = None
    if train_subset_is_limited:
        val_sequences_per_epoch = min(len(val_ds), max(1, math.ceil(train_sequences_per_epoch * 0.1)))

    train_sampler = EpochSubsetSampler(
        len(train_ds),
        max_samples=train_sequences_per_epoch,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    val_sampler = EpochSubsetSampler(
        len(val_ds),
        max_samples=val_sequences_per_epoch,
        shuffle=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
        persistent_workers=args.num_workers > 0,
        
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
        persistent_workers=args.num_workers > 0,
    )

    log.info(
        "Train: %d/%d sequences per epoch (%d batches, shuffle=%s)",
        len(train_sampler), len(train_ds), len(train_loader), args.shuffle,
    )
    if train_subset_is_limited:
        log.info(
            "Train subset selection: %s",
            "reshuffled each epoch" if args.shuffle else "fixed deterministic subset",
        )
        log.info(
            "Val:   %d/%d sequences per epoch (%d batches, 10%% of capped train)",
            len(val_sampler), len(val_ds), len(val_loader),
        )
    else:
        log.info("Val:   %d sequences (%d batches)", len(val_ds), len(val_loader))
    log.info("Embedding width: %d", train_ds.embed_dim)
    if args.multiclass:
        log.info("Multiclass feature model: %s", train_ds.feature_model_name)

    # ── Model ─────────────────────────────────────────────────────────────
    model_kwargs = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "dropout": args.dropout,
        "embed_dim": train_ds.embed_dim,
        "flow_mode": args.flow_mode,
        "flow_dense_size": args.flow_dense_size,
        "use_aux_layers": args.use_aux_layers,
    }
    if args.multiclass:
        model_kwargs.update({
            "n_partners": args.n_partners,
            "n_beholders": args.n_beholders,
            "n_beholder_keypoints": args.n_beholder_keypoints,
        })
    if args.use_kinematics:
        model_kwargs["use_kinematics"] = True
        model_kwargs["kin_dim"] = args.kin_dim
    if args.use_ddl:
        model_kwargs["use_ddl"] = True
    if args.use_gated_fusion:
        model_kwargs["use_gated_fusion"] = True
    if args.multiclass and not args.no_difference_pathway:
        model_kwargs["use_difference_pathway"] = True
        model_kwargs["difference_dim"] = args.difference_dim

    model = FunscriptTCN(**model_kwargs).to(device)

    # checkpoint = Path("data\\models\\checkpoints_tcn\\tcn_epoch500.pt")
    # if checkpoint.exists():
    #     log.info("Loading checkpoint from %s", checkpoint)
    #     ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    #     model.load_state_dict(ckpt["model_state_dict"])

    # Resume from checkpoint (for multi-phase training)
    if args.resume is not None:
        log.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Loaded model (epoch %s, val_loss=%s)",
                 ckpt.get("epoch", "?"), f"{ckpt.get('val_loss', 0):.6f}")
        if not args.use_aux_layers:
            log.info("  Disabling auxiliary layers as per command-line argument")
            model.disable_aux_layers()
                
            

    params = model.count_parameters()
    log.info("Model: %s trainable / %s total parameters",
             f"{params['trainable']:,}", f"{params['total']:,}")

    # ── Phase-specific freezing ───────────────────────────────────────────
    if args.phase is not None:
        _apply_phase_freezing(model, args.phase, args.fill_with_noise)
        params = model.count_parameters()
        log.info("Phase %d: %s trainable / %s total parameters",
                 args.phase, f"{params['trainable']:,}", f"{params['total']:,}")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
            eta_min=args.lr * 0.01,
        )
    elif args.scheduler == "CosineWarmupLR":
        restart_period = args.epochs * len(train_loader) // 4
        restart_period = max(restart_period, 1)  # avoid zero or negative period
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_period, T_mult=1, eta_min=args.lr * 0.01
        )
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.99,
            patience=10,
            min_lr=args.lr * 0.001,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")


    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def merge_model(
        model: FunscriptTCN,
        flow_encoder_path: Path | str,
        pose_encoder_path: Path | str,
        emb_encoder_path: Path | str,
    ) -> None:
        """Merge pretrained encoder weights into the current model."""
        flow_model = FunscriptTCN(**model_kwargs).to(device)
        pose_model = FunscriptTCN(**model_kwargs).to(device)
        emb_model = FunscriptTCN(**model_kwargs).to(device)

        flow_ckpt = torch.load(flow_encoder_path, map_location=device, weights_only=False)
        pose_ckpt = torch.load(pose_encoder_path, map_location=device, weights_only=False)
        emb_ckpt = torch.load(emb_encoder_path, map_location=device, weights_only=False)
        flow_model.load_state_dict(flow_ckpt["model_state_dict"])
        pose_model.load_state_dict(pose_ckpt["model_state_dict"])
        emb_model.load_state_dict(emb_ckpt["model_state_dict"])

        # Copy flow encoder weights
        model.flow_encoder.load_state_dict(flow_model.flow_encoder.state_dict()) 
        # Copy pose encoder weights
        model.pose_encoder.load_state_dict(pose_model.pose_encoder.state_dict())
        # Copy embedding encoder weights
        model.emb_encoder.load_state_dict(emb_model.emb_encoder.state_dict())
        
        # freeze encoders after merging
        for p in model.flow_encoder.parameters():
            p.requires_grad = False
        for p in model.pose_encoder.parameters():
            p.requires_grad = False
        for p in model.emb_encoder.parameters():
            p.requires_grad = False

        return model

    flow_encoder_model = "data\\models\\checkpoints_tcn\\phase5_flow_best_tcn.pt"
    pose_encoder_model = "data\\models\\checkpoints_tcn\\phase6_emb_best_tcn.pt"
    emb_encoder_model = "data\\models\\checkpoints_tcn\\phase7_pose_best_tcn.pt"
    # test merge currently not used.
    #model = merge_model(model, flow_encoder_model, pose_encoder_model, emb_encoder_model)

    model.eval()
    val_losses = []
    val_pred_means = []
    val_pred_stds = []
    val_metric_history = {
        "pos_mse": [],
        "event_mse": [],
        "active_mse": [],
        "vel_mse": [],
        "vel_mae": [],
        "acc_mse": [],
        "acc_mae": [],
        "spec_mse": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            kp = batch["keypoints"].to(device, non_blocking=True)
            emb = batch["embeddings"].to(device, non_blocking=True)
            fl = batch["flow"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(kp, emb, fl)  # [B, 4, T]
                loss, main_pred, metric_batch = _compute_loss_from_multichannel(pred, lbl, args)

            val_losses.append(loss.item())
            for key in val_metric_history:
                if key in metric_batch:
                    val_metric_history[key].append(metric_batch[key].item())
            val_pred_means.append(main_pred.mean().item())
            val_pred_stds.append(main_pred.std().item())

    avg_val = np.mean(val_losses)
    avg_val_metrics = {key: float(np.mean(values)) for key, values in val_metric_history.items()}
    avg_pred_mean = np.mean(val_pred_means)
    avg_pred_std = np.mean(val_pred_stds)
    log.info("Initial validation loss: %.6f", avg_val)
    log.info("Initial validation metrics: %s", ", ".join(f"{k}={v:.6f}" for k, v in avg_val_metrics.items()))
    log.info("Initial validation prediction mean: %.6f, std: %.6f", avg_pred_mean, avg_pred_std)

    #best_val_loss = float("inf")
    best_val_loss = avg_val
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr
    # ── Logging / Checkpoints ─────────────────────────────────────────────
    run_name = f"tcn_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    checkpoint_dir = Path("data/models/checkpoints_tcn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config = extract_model_config({
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "kernel_size": 3,
        "dropout": args.dropout,
        "n_keypoints": 21,
        "embed_dim": train_ds.embed_dim,
        "flow_dim": 64,
        "flow_mode": args.flow_mode,
        "flow_dense_size": args.flow_dense_size,
    })
    if args.multiclass:
        model_config.update({
            "n_partners": args.n_partners,
            "n_beholders": args.n_beholders,
            "n_beholder_keypoints": args.n_beholder_keypoints,
        })
    else:
        model_config["n_persons"] = 10
    if args.use_kinematics:
        model_config["use_kinematics"] = True
        model_config["kin_dim"] = args.kin_dim
    if args.use_ddl:
        model_config["use_ddl"] = True
    if args.use_gated_fusion:
        model_config["use_gated_fusion"] = True
    if args.multiclass and not args.no_difference_pathway:
        model_config["use_difference_pathway"] = True
        model_config["difference_dim"] = args.difference_dim

    data_config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "shuffle": args.shuffle,
        "train_sequences_per_epoch": len(train_sampler),
        "val_sequences_per_epoch": len(val_sampler),
        "stats_path": str(train_ds.stats_path) if train_ds.stats_path is not None else None,
        "feature_model_name": train_ds.feature_model_name if args.multiclass else None,
    }
    metric_config = {
        "event_weight": args.event_weight,
        "event_activity_gain": args.event_activity_gain,
        "event_activity_power": args.event_activity_power,
        "active_quantile": args.active_quantile,
        "temporal_weight": args.temporal_weight,
        "velocity_weight": args.velocity_weight,
        "spectral_weight": args.spectral_weight,
        "spectral_kernel": args.spectral_kernel,
        "aux_weight": args.aux_weight,
    }

    log.info("Run dir: %s", run_dir)
    log.info("Checkpoint dir: %s", checkpoint_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    _early_stop_counter = 0
    original_best_val_loss = best_val_loss
    improved = False

    #for epoch in range(1, args.epochs + 1):
    epoch = 0
    epochs = args.epochs
    overtime = False

    while epoch <= epochs:
        epoch += 1 
        epoch_start = time.time()

        # --- Train ---
        train_sampler.set_epoch(epoch - 1)
        model.train()
        train_losses = []
        train_metric_history = {
            "pos_mse": [],
            "event_mse": [],
            "active_mse": [],
            "vel_mse": [],
            "vel_mae": [],
            "acc_mse": [],
            "acc_mae": [],
            "spec_mse": [],
        }

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
            kp = batch["keypoints"].to(device, non_blocking=True)
            emb = batch["embeddings"].to(device, non_blocking=True)
            fl = batch["flow"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(kp, emb, fl)  # [B, 4, T]
                loss, main_pred, metric_batch = _compute_loss_from_multichannel(pred, lbl, args)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler != "ReduceLROnPlateau":
                scheduler.step()

            train_losses.append(loss.item())
            for key in train_metric_history:
                if key in metric_batch:
                    train_metric_history[key].append(metric_batch[key].item())
            global_step += 1

            if global_step % 50 == 0:
                pos_loss = ((1.0 - args.event_weight) * metric_batch["pos_mse"]
                            + args.event_weight * metric_batch["event_mse"])
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/pos_loss", pos_loss.item(), global_step)
                writer.add_scalar("train/pos_mse", metric_batch["pos_mse"].item(), global_step)
                writer.add_scalar("train/event_mse", metric_batch["event_mse"].item(), global_step)
                writer.add_scalar("train/active_mse", metric_batch["active_mse"].item(), global_step)
                writer.add_scalar("train/vel_mae", metric_batch["vel_mae"].item(), global_step)
                writer.add_scalar("train/acc_mae", metric_batch["acc_mae"].item(), global_step)
                writer.add_scalar("train/pred_mean", main_pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", main_pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                if args.use_aux_layers:
                    if "aux_loss" in metric_batch:
                        writer.add_scalar("train/aux_loss", metric_batch["aux_loss"].item(), global_step)

        avg_train = np.mean(train_losses)
        avg_train_metrics = {key: float(np.mean(values)) for key, values in train_metric_history.items()}

        # --- Validate ---
        model.eval()
        val_losses = []
        val_pred_means = []
        val_pred_stds = []
        val_metric_history = {
            "pos_mse": [],
            "event_mse": [],
            "active_mse": [],
            "vel_mse": [],
            "vel_mae": [],
            "acc_mse": [],
            "acc_mae": [],
            "spec_mse": [],
        }
        # check if it's the last epoch to set the flag for dataset to disable augmentations if needed
        if epoch == args.epochs:
            # reinit val loader with last_epoch=True to disable augmentations if dataset is designed that way
            val_ds.last_epoch = True
            del val_loader
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
                persistent_workers=args.num_workers > 0,
            )
            

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
                kp = batch["keypoints"].to(device, non_blocking=True)
                emb = batch["embeddings"].to(device, non_blocking=True)
                fl = batch["flow"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(kp, emb, fl)  # [B, 4, T]
                    loss, main_pred, metric_batch = _compute_loss_from_multichannel(pred, lbl, args)

                val_losses.append(loss.item())
                for key in val_metric_history:
                    if key in metric_batch:
                        val_metric_history[key].append(metric_batch[key].item())
                val_pred_means.append(main_pred.mean().item())
                val_pred_stds.append(main_pred.std().item())

        avg_val = np.mean(val_losses)
        avg_val_metrics = {key: float(np.mean(values)) for key, values in val_metric_history.items()}
        avg_pred_mean = np.mean(val_pred_means)
        avg_pred_std = np.mean(val_pred_stds)

        epoch_time = time.time() - epoch_start

        log.info(
            "Epoch %3d/%d | train=%.6f val=%.6f pos=%.6f event=%.6f active=%.6f vel_mae=%.6f | "
            "pred_μ=%.3f pred_σ=%.3f | lr=%.2e | %.1fs",
            epoch, args.epochs, avg_train, avg_val, avg_val_metrics["pos_mse"],
            avg_val_metrics["event_mse"], avg_val_metrics["active_mse"], avg_val_metrics["vel_mae"],
            avg_pred_mean, avg_pred_std,
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        writer.add_scalar("val/loss", avg_val, epoch)
        writer.add_scalar("val/pos_loss", avg_val_metrics["pos_mse"], epoch)
        writer.add_scalar("val/event_mse", avg_val_metrics["event_mse"], epoch)
        writer.add_scalar("val/active_mse", avg_val_metrics["active_mse"], epoch)
        writer.add_scalar("val/vel_mae", avg_val_metrics["vel_mae"], epoch)
        writer.add_scalar("val/acc_mae", avg_val_metrics["acc_mae"], epoch)
        writer.add_scalar("val/pred_mean", avg_pred_mean, epoch)
        writer.add_scalar("val/pred_std", avg_pred_std, epoch)
        writer.add_scalar("train/loss_epoch", avg_train, epoch)
        writer.add_scalar("train/pos_loss_epoch", avg_train_metrics["pos_mse"], epoch)
        writer.add_scalar("train/event_mse_epoch", avg_train_metrics["event_mse"], epoch)
        writer.add_scalar("train/active_mse_epoch", avg_train_metrics["active_mse"], epoch)

        # Update augmentation scale based on val/train loss ratio
        augment_scale = _compute_augment_scale(avg_val, avg_train)
        train_ds.augment_scale = augment_scale
        writer.add_scalar("train/augment_scale", augment_scale, epoch)
        log.info("  Augment scale: %.3f (val/train ratio=%.3f)", augment_scale, avg_val / max(avg_train, 1e-9))

        if args.scheduler == "ReduceLROnPlateau":
            scheduler.step(avg_val)

        # --- Prediction overlay plots every 10 epochs (like old train.py) ---
        if epoch % 2 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # samples = val_ds.sample_high_variance(n=8, device=device)
                if epoch == 1: # choose 4 random, and 4 high-variance samples in the first epoch and reuse them for consistency in future epochs
                    samples = val_ds.sample_random(n=4, device=device)
                    samples_high_var = val_ds.sample_high_variance(n=4, device=device)
                    samples = {key: torch.cat([samples[key], samples_high_var[key]], dim=0) for key in samples}
                kp_s = samples["keypoints"]
                emb_s = samples["embeddings"]
                fl_s = samples["flow"]
                lbl_s = samples["labels"]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred_s = model(kp_s, emb_s, fl_s)  # [8, 4, T]

                pred_s = pred_s.float().cpu().numpy()  # [8, 4, T]
                lbl_s = lbl_s.float().cpu().numpy()

            n_plots = min(8, len(pred_s))
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=False)
            if n_plots == 1:
                axes = [axes]
            if args.use_aux_layers:
                aux_colors = ["green", "purple", "red"]
                aux_labels = ["pose", "emb", "flow"]
                for i, ax in enumerate(axes):
                    ax.plot(lbl_s[i], label="target", alpha=0.85, lw=1.5, color="steelblue")
                    ax.plot(pred_s[i, 0], label="fused", alpha=0.85, lw=1.5, color="darkorange")
                    for ch in range(1, 4):
                        ax.plot(pred_s[i, ch], label=aux_labels[ch - 1],
                                alpha=0.4, lw=0.8, color=aux_colors[ch - 1])
                    
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_ylabel(f"#{i}", fontsize=7)
                    if i == 0:
                        ax.legend(fontsize=7)
                        ax.set_title(f"Epoch {epoch} predictions (high-variance val samples)")
            else:
                # just plot the main prediction channel without aux branches
                for i, ax in enumerate(axes):
                    ax.plot(lbl_s[i], label="target", alpha=0.85, lw=1.5, color="steelblue")
                    ax.plot(pred_s[i, 0], label="pred", alpha=0.85, lw=1.5, color="darkorange")
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_ylabel(f"#{i}", fontsize=7)
                    if i == 0:
                        ax.legend(fontsize=7)
                        ax.set_title(f"Epoch {epoch} predictions (high-variance val samples)")


                
            fig.tight_layout()
            writer.add_figure("Predictions/overlay", fig, epoch)
            plt.close(fig)

        # --- Checkpoint ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": avg_val,
                "global_step": global_step,
                "model_config": model_config,
                "data_config": data_config,
                "metric_config": metric_config,
            }, checkpoint_dir / "best_tcn.pt")
            log.info("  → New best val loss: %.6f", avg_val)
        else:
            _early_stop_counter += 1
            if args.early_stopping_patience > 0 and _early_stop_counter >= args.early_stopping_patience:
                log.info(
                    "Early stopping: no improvement for %d epochs (best val=%.6f)",
                    _early_stop_counter, best_val_loss,
                )
                break
                #_early_stop_counter = 0
                # reusing this for model blending with best val model, since we want to continue training after early stopping anyway
                # best_model_path = checkpoint_dir / "best_tcn.pt"
                # best_model = FunscriptTCN(**model_config).to(device)
                # if best_model_path.exists():
                #     log.info("Loading best model from %s for continued training", best_model_path)
                #     ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
                #     best_model.load_state_dict(ckpt["model_state_dict"])
                #     # blend weights onto the current model (simple moving average with blending factor)
                #     blending_factor = 0.5
                #     with torch.no_grad():
                #         for p, best_p in zip(model.parameters(), best_model.parameters()):
                #             p.data = blending_factor * p.data + (1 - blending_factor) * best_p.data
                            

                # else:
                #     log.warning("Best model checkpoint not found at %s — cannot load for continued training", best_model_path)
                #     break
                

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "model_config": model_config,
                "global_step": global_step,
                "data_config": data_config,
                "metric_config": metric_config,
            }, checkpoint_dir / f"tcn_epoch{epoch}.pt")

        if best_val_loss < original_best_val_loss:
            improved = True
        if epoch == 8 and not improved:
            break
        # if it's the last epoch and we have improved in the last 3, switch to ReduceLROnPlateau scheduler until and keep going until early stopping patience exceeds 3
        if epoch == args.epochs and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and improved:
            log.info("Switching to ReduceLROnPlateau scheduler for fine-tuning")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=args.lr * 0.0001,
            )
            epochs += 100  # extend training for fine-tuning
            args.scheduler = "ReduceLROnPlateau"
            overtime = True
        if overtime and _early_stop_counter >= 3:
            log.info("Early stopping during fine-tuning: no improvement for 3 epochs")
            break
        


    writer.close()
    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    train()
