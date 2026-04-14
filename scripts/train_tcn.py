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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Resolve imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.tcn import FunscriptTCN, extract_model_config
from src.training.funscript_metrics import compute_regression_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

log = logging.getLogger(__name__)


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

    def __init__(
        self,
        data_dir: Path,
        split: str,
        seq_len: int = 120,
        stride: int = 60,
        n_persons: int = 10,
        n_keypoints: int = 21,
        embed_dim: int = 512,
        flow_dim: int = 64,
        augment: bool = False,
        multiclass: bool = False,
        phase: int = -1,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.n_persons = n_persons
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim
        self.augment = augment
        self.multiclass = multiclass
        self.phase = phase
        self.using_stats = False
        self.stats_path = None
        self.last_epoch = False

        if multiclass:
            self.KP_FILE = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
            self.EMB_FILE = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
            self.EMB_FILE_META = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.json"
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

            if not all(p.exists() for p in [kp_path, emb_path, emb_meta_path, flow_path, label_path]):
                skipped += 1
                continue
            
            # check for review.json in scene directory and skip if "status": "rejected" or "stage2_status": "rejected"
            review_path = vid_dir / "review.json"
            if review_path.exists():
                review = json.loads(review_path.read_text(encoding="utf-8"))
                if review.get("status") == "rejected" or review.get("stage2_status") == "rejected":
                    skipped += 1
                    continue
            # check embedding metadata is valid for current extraction method
            if emb_meta_path.exists():
                meta = json.loads(emb_meta_path.read_text(encoding="utf-8"))
                if meta.get("method") != "single_pass_hook_roi_align":
                    skipped += 1
                    continue
                if self.multiclass and not meta.get("multiclass", False):
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

    def _load_stats(self) -> None:
        stats_path = self.data_dir / "featurestats" / "feature_stats.npz"
        self.using_stats = False
        if not stats_path.exists():
            log.warning("No feature_stats.npz — features will NOT be normalized")
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

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        vid_dir = self.data_dir / "processed" / vid_id
        end = start + self.seq_len

        labels = np.load(str(vid_dir / "labels.npy"), mmap_mode="r")[start:end].copy()
        keypoints = np.load(str(vid_dir / self.KP_FILE), mmap_mode="r")[start:end].copy()
        embeddings = np.load(str(vid_dir / self.EMB_FILE), mmap_mode="r")[start:end].copy()
        flow = np.load(str(vid_dir / self.FLOW_FILE), mmap_mode="r")[start:end].copy()

        # Normalize
        if self.emb_mean is not None:
            embeddings = (embeddings - self.emb_mean) / (self.emb_std + 1e-8)
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



        # s = self.augment_scale  # [0, 1] — 0=min values, 1=max values

        # dropout_prob = 0.1 + s * (0.9 - 0.1)
        # if torch.rand(1).item() < dropout_prob:
        #     choice = torch.randint(3, (1,)).item()
        #     if choice == 0:
        #         kp = torch.zeros_like(kp)
        #     elif choice == 1:
        #         emb = torch.zeros_like(emb)
        #     else:
        #         flow = torch.zeros_like(flow)

        # emb_noise_prob = 0.2 + s * (0.9 - 0.2)    
        # emb_noise_mag  = 0.02 + s * (0.15 - 0.02)
        # if torch.rand(1).item() < emb_noise_prob:
        #     emb = emb + torch.randn_like(emb) * emb_noise_mag

        # flow_noise_prob = 0.2 + s * (0.9 - 0.2) 
        # flow_noise_mag  = 0.02 + s * (0.2 - 0.02)
        # if torch.rand(1).item() < flow_noise_prob:
        #     # flow = flow + torch.randn_like(flow) * flow_noise_mag # makes B x Seq x FlowDim noise which may be too random
        #     # Alternate: use same noise for all flow frames in a sequence to generalize better to different flow magnitudes
        #     noise = torch.randn(flow.shape[1], device=flow.device) * flow_noise_mag
        #     flow = flow + noise


        # kp_noise_prob = 0.2 + s * (0.3 - 0.2)
        # kp_noise_mag = 0.01 + s * (0.04 - 0.01) # 5% max jitter
        # if torch.rand(1).item() < kp_noise_prob:
        #     kp = kp + torch.randn_like(kp) * kp_noise_mag
        #     kp = torch.clamp(kp, 0.0, 1.0)
        #     # decay all kp confidences by half
        #     kp[..., 2] = kp[..., 2] * 0.5

        # kp_intermittent_drop_prob = 0.2 + s * (0.7 - 0.2)
        # if torch.rand(1).item() < kp_intermittent_drop_prob:
        #     # Randomly zero out all keypoints for random contiguous segments (simulate occlusion)
        #     T = kp.shape[0]
        #     n_segments = max(1, int(T * 0.01))  # number of segments scales with sequence length
        #     for _ in range(n_segments):
        #         seg_len = torch.randint(5, 20, (1,)).item()  # segment length between 5 and 20 frames
        #         start = torch.randint(0, T - seg_len, (1,)).item()
        #         kp[start:start + seg_len] = 0.0



        return kp, emb, flow, labels


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
    import random
    parser = argparse.ArgumentParser(description="Train TCN funscript model")
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="CosineWarmupLR",
                        choices=["OneCycleLR", "CosineAnnealingLR", "CosineWarmupLR", "ReduceLROnPlateau"])
    parser.add_argument("--multiclass", action="store_true",
                        help="Use multiclass model (partner + beholder)")
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
    parser.add_argument("--velocity-weight", type=float, default=0.0,
                        help="Weight for velocity-matching loss (first derivative)")
    parser.add_argument("--spectral-weight", type=float, default=0.0,
                        help="Weight for high-frequency detail loss (multi-scale spectral)")
    parser.add_argument("--spectral-kernel", type=int, default=15,
                        help="Moving-average kernel size for spectral loss low-pass filter")
    parser.add_argument("--event-weight", type=float, default=0.25,
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
    args = parser.parse_args()

    if not 0.0 <= args.event_weight <= 1.0:
        parser.error("--event-weight must be between 0 and 1")
    if not 0.0 < args.active_quantile <= 1.0:
        parser.error("--active-quantile must be in (0, 1]")

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
        n_persons=n_total, augment=True, multiclass=args.multiclass, phase=args.phase
    )
    val_ds = MotionDataset(
        args.data_dir, "val", args.seq_len, args.stride,
        n_persons=n_total, augment=False, multiclass=args.multiclass
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
        persistent_workers=args.num_workers > 0,
        
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
        persistent_workers=args.num_workers > 0,
    )

    log.info("Train: %d sequences (%d batches)", len(train_ds), len(train_loader))
    log.info("Val:   %d sequences (%d batches)", len(val_ds), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model_kwargs = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "dropout": args.dropout,
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

    # if checkpoint.exists():
    #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    #     scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    #     log.info("Loaded optimizer and scheduler state from checkpoint")

    best_val_loss = float("inf")
    if args.resume is not None:
        # if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
        #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        #     #scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        #     log.info("Loaded optimizer and scheduler state from checkpoint")
        #     if args.lr is not None:
        #         # If resuming but a new lr is specified, override the loaded scheduler state to use the new lr
        #         for param_group in optimizer.param_groups:
        #             param_group["lr"] = args.lr
        #         log.info("Overriding loaded learning rate with new value: %s", args.lr)
        # else:
        #     log.warning("No optimizer/scheduler state found in checkpoint — starting with fresh optimizer/scheduler")
        if "val_loss" in ckpt and args.load_best_val_loss:
            best_val_loss = ckpt["val_loss"]
            log.info("Resuming with best_val_loss = %.6f", best_val_loss)
        else:
            # run an initial validation loop to get the current val loss for early stopping
            log.info("No val_loss found in checkpoint — running initial validation to get baseline val loss for early stopping")
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

            with torch.no_grad():
                for batch in val_loader:
                    kp = batch["keypoints"].to(device, non_blocking=True)
                    emb = batch["embeddings"].to(device, non_blocking=True)
                    fl = batch["flow"].to(device, non_blocking=True)
                    lbl = batch["labels"].to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        pred = model(kp, emb, fl)
                        metric_batch = compute_regression_metrics(
                            pred,
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

                    val_losses.append(loss.item())
                    for key in val_metric_history:
                        val_metric_history[key].append(metric_batch[key].item())
                    val_pred_means.append(pred.mean().item())
                    val_pred_stds.append(pred.std().item())

            avg_val = np.mean(val_losses)
            avg_val_metrics = {key: float(np.mean(values)) for key, values in val_metric_history.items()}
            avg_pred_mean = np.mean(val_pred_means)
            avg_pred_std = np.mean(val_pred_stds)
            log.info("Initial validation loss: %.6f", avg_val)
            log.info("Initial validation metrics: %s", ", ".join(f"{k}={v:.6f}" for k, v in avg_val_metrics.items()))
            log.info("Initial validation prediction mean: %.6f, std: %.6f", avg_pred_mean, avg_pred_std)
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
        "embed_dim": 512,
        "flow_dim": 64,
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
        "stats_path": str(train_ds.stats_path) if train_ds.stats_path is not None else None,
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
    }

    log.info("Run dir: %s", run_dir)
    log.info("Checkpoint dir: %s", checkpoint_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    _early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Train ---
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

        for batch in train_loader:
            kp = batch["keypoints"].to(device, non_blocking=True)
            emb = batch["embeddings"].to(device, non_blocking=True)
            fl = batch["flow"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(kp, emb, fl)  # [B, T]
                metric_batch = compute_regression_metrics(
                    pred,
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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler != "ReduceLROnPlateau":
                scheduler.step()

            train_losses.append(loss.item())
            for key in train_metric_history:
                train_metric_history[key].append(metric_batch[key].item())
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/pos_loss", pos_loss.item(), global_step)
                writer.add_scalar("train/pos_mse", metric_batch["pos_mse"].item(), global_step)
                writer.add_scalar("train/event_mse", metric_batch["event_mse"].item(), global_step)
                writer.add_scalar("train/active_mse", metric_batch["active_mse"].item(), global_step)
                writer.add_scalar("train/vel_mae", metric_batch["vel_mae"].item(), global_step)
                writer.add_scalar("train/acc_mae", metric_batch["acc_mae"].item(), global_step)
                writer.add_scalar("train/temp_loss", temp_loss.item(), global_step)
                writer.add_scalar("train/vel_loss", vel_loss.item(), global_step)
                writer.add_scalar("train/spec_loss", spec_loss.item(), global_step)
                writer.add_scalar("train/pred_mean", pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

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
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
                persistent_workers=args.num_workers > 0,
            )
            

        with torch.no_grad():
            for batch in val_loader:
                kp = batch["keypoints"].to(device, non_blocking=True)
                emb = batch["embeddings"].to(device, non_blocking=True)
                fl = batch["flow"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(kp, emb, fl)
                    metric_batch = compute_regression_metrics(
                        pred,
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

                val_losses.append(loss.item())
                for key in val_metric_history:
                    val_metric_history[key].append(metric_batch[key].item())
                val_pred_means.append(pred.mean().item())
                val_pred_stds.append(pred.std().item())

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
                samples = val_ds.sample_high_variance(n=8, device=device)
                kp_s = samples["keypoints"]
                emb_s = samples["embeddings"]
                fl_s = samples["flow"]
                lbl_s = samples["labels"]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred_s = model(kp_s, emb_s, fl_s)  # [8, T]

                pred_s = pred_s.float().cpu().numpy()
                lbl_s = lbl_s.float().cpu().numpy()

            n_plots = min(8, len(pred_s))
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=False)
            if n_plots == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                ax.plot(lbl_s[i], label="target", alpha=0.85, lw=1.5, color="steelblue")
                ax.plot(pred_s[i], label="pred",   alpha=0.85, lw=1.5, color="darkorange")
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
                _early_stop_counter = 0
                # reusing this for model blending with best val model, since we want to continue training after early stopping anyway
                best_model_path = checkpoint_dir / "best_tcn.pt"
                best_model = FunscriptTCN(**model_config).to(device)
                if best_model_path.exists():
                    log.info("Loading best model from %s for continued training", best_model_path)
                    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
                    best_model.load_state_dict(ckpt["model_state_dict"])
                    # blend weights onto the current model (simple moving average with blending factor)
                    blending_factor = 0.5
                    with torch.no_grad():
                        for p, best_p in zip(model.parameters(), best_model.parameters()):
                            p.data = blending_factor * p.data + (1 - blending_factor) * best_p.data
                            

                else:
                    log.warning("Best model checkpoint not found at %s — cannot load for continued training", best_model_path)
                    break
                

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

    writer.close()
    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    train()
