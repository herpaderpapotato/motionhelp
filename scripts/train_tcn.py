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
from src.models.tcn import FunscriptTCN

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
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.n_persons = n_persons
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim
        self.augment = augment
        self.multiclass = multiclass

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
        stats_path = self.data_dir / "feature_stats.npz"
        if not stats_path.exists():
            log.warning("No feature_stats.npz — features will NOT be normalized")
            self.emb_mean = self.emb_std = None
            self.flow_mean = self.flow_std = None
            return

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

        if self.augment:
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
        # Time reversal (50%)
        if torch.rand(1).item() < 0.5:
            kp = kp.flip(0)
            emb = emb.flip(0)
            flow = -flow.flip(0)
            labels = labels.flip(0)
        elif torch.rand(1).item() < 0.1:
            # zero out a random input, kp, emb, or flow (but not labels)
            choice = torch.randint(3, (1,)).item()
            if choice == 0:
                kp = torch.zeros_like(kp)
            elif choice == 1:
                emb = torch.zeros_like(emb)
            else:
                flow = torch.zeros_like(flow)
        elif torch.rand(1).item() < 0.1:
            # Add small noise to embeddings or flow
            if torch.rand(1).item() < 0.5:
                emb += torch.randn_like(emb) * 0.02
            else:
                flow += torch.randn_like(flow) * 0.02
        return kp, emb, flow, labels


# ---------------------------------------------------------------------------
# Phase-specific freezing for multi-phase training
# ---------------------------------------------------------------------------

def _apply_phase_freezing(model: FunscriptTCN, phase: int) -> None:
    """Freeze/unfreeze model parameters based on training phase.

    Phase 1: Normal full training (all params trainable)
    Phase 2: Train only pose pathways (freeze emb, flow, TCN, output)
    Phase 3: Train only embedding pathways (freeze pose, flow, TCN, output)
    Phase 4: Train only flow pathway (freeze pose, emb, TCN, output)
    5 flow encoder + fusion + output head
    6 embeddings + fusion + output head
    7 pose pathways + fusion + output head
    8 flow encoder, embedding encoders, fusion + output head + TCN backbone
    9 fusion + output head + TCN backbone
    10  only pose pathways
    11 fusion + output head + TCN backbone
    12 phase 1 again (but do it at a low lr)

    So:
    We get it to a reasonable state.
    We make sure we're doing the best we can with pose.
    We make sure we're doing the best we can with embedding
    We make sure we're doing the best we can with flow
    We give flow a chance to dominate the path
    We give embeddings a chance to dominate the path
    We give pose a chance to dominate the path
    We optimize the path again
    We just tune pose for the path.
    We optimize the path again
    We make extremely small changes to the whole thing.

    Ideally I'd also zero out the data or handle the augmentation differently. I probably do need to stop augmentation zeroing out the primary input in that scenario.
    """
    # First, freeze everything
    if phase > 1:
        for p in model.parameters():
            p.requires_grad = False

    if phase == 1:
        pass  # all trainable
    elif phase == 2:
        # Unfreeze pose encoders only
        for p in model.pose_encoder.parameters():
            p.requires_grad = True
        for p in model.pose_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_pose_encoder.parameters():
                p.requires_grad = True
    elif phase == 3:
        # Unfreeze embedding encoders only
        for p in model.emb_encoder.parameters():
            p.requires_grad = True
        for p in model.emb_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_emb_encoder.parameters():
                p.requires_grad = True
    elif phase == 4:
        # Unfreeze flow encoder only
        for p in model.flow_encoder.parameters():
            p.requires_grad = True
    elif phase == 5:
        # Unfreeze flow encoder, fusion + output head + TCN backbone
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
        # Unfreeze fusion + output head + TCN backbone
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 10:
        # Unfreeze pose encoders only
        for p in model.pose_encoder.parameters():
            p.requires_grad = True
        for p in model.pose_attn.parameters():
            p.requires_grad = True
        if model.multiclass:
            for p in model.beholder_pose_encoder.parameters():
                p.requires_grad = True
    elif phase == 11:
        # Unfreeze fusion + output head + TCN backbone
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.output_head.parameters():
            p.requires_grad = True
        for p in model.tcn_blocks.parameters():
            p.requires_grad = True
    elif phase == 12:
        pass  # all trainable
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1-12.")

    log.info("Phase %d freezing applied", phase)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="OneCycleLR",
                        choices=["OneCycleLR", "CosineAnnealingLR", "CosineWarmupLR", "ReduceLROnPlateau"])
    parser.add_argument("--multiclass", action="store_true",
                        help="Use multiclass model (partner + beholder)")
    parser.add_argument("--n-partners", type=int, default=5)
    parser.add_argument("--n-beholders", type=int, default=1)
    parser.add_argument("--n-beholder-keypoints", type=int, default=7)
    parser.add_argument("--phase", type=int, default=None,
                        help="Training phase (1-6) for multi-phase training")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint (for multi-phase training)")
    parser.add_argument("--early-stopping-patience", type=int, default=50,
                        help="Stop training if val loss has not improved for this many epochs (0 = disabled)")
    args = parser.parse_args()

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
        torch.cuda.manual_seed_all(args.seed)

    # ── Datasets ──────────────────────────────────────────────────────────
    n_total = (args.n_partners + args.n_beholders) if args.multiclass else 10

    train_ds = MotionDataset(
        args.data_dir, "train", args.seq_len, args.stride,
        n_persons=n_total, augment=True, multiclass=args.multiclass,
    )
    val_ds = MotionDataset(
        args.data_dir, "val", args.seq_len, args.stride,
        n_persons=n_total, augment=False, multiclass=args.multiclass,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
        _apply_phase_freezing(model, args.phase)
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


    # ── Logging / Checkpoints ─────────────────────────────────────────────
    run_name = f"tcn_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    checkpoint_dir = Path("data/models/checkpoints_tcn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "kernel_size": 3,
        "dropout": args.dropout,
        "n_keypoints": 21,
        "embed_dim": 512,
        "flow_dim": 64,
    }
    if args.multiclass:
        model_config.update({
            "n_partners": args.n_partners,
            "n_beholders": args.n_beholders,
            "n_beholder_keypoints": args.n_beholder_keypoints,
        })
    else:
        model_config["n_persons"] = 10

    log.info("Run dir: %s", run_dir)
    log.info("Checkpoint dir: %s", checkpoint_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step = 0
    _early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        train_pos_losses = []

        for batch in train_loader:
            kp = batch["keypoints"].to(device, non_blocking=True)
            emb = batch["embeddings"].to(device, non_blocking=True)
            fl = batch["flow"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(kp, emb, fl)  # [B, T]

                pos_loss = nn.functional.mse_loss(pred, lbl)

                if args.temporal_weight > 0 and pred.shape[1] > 2:
                    pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
                    tgt_acc = lbl[:, 2:] - 2 * lbl[:, 1:-1] + lbl[:, :-2]
                    temp_loss = nn.functional.mse_loss(pred_acc, tgt_acc)
                else:
                    temp_loss = torch.tensor(0.0, device=device)

                loss = pos_loss + args.temporal_weight * temp_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler == "ReduceLROnPlateau":
                scheduler.step(loss)
            else:
                scheduler.step()

            train_losses.append(loss.item())
            train_pos_losses.append(pos_loss.item())
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/pos_loss", pos_loss.item(), global_step)
                writer.add_scalar("train/temp_loss", temp_loss.item(), global_step)
                writer.add_scalar("train/pred_mean", pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

        avg_train = np.mean(train_losses)
        avg_train_pos = np.mean(train_pos_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        val_pos_losses = []
        val_pred_means = []
        val_pred_stds = []

        with torch.no_grad():
            for batch in val_loader:
                kp = batch["keypoints"].to(device, non_blocking=True)
                emb = batch["embeddings"].to(device, non_blocking=True)
                fl = batch["flow"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(kp, emb, fl)
                    pos_loss = nn.functional.mse_loss(pred, lbl)

                    if pred.shape[1] > 2:
                        pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
                        tgt_acc = lbl[:, 2:] - 2 * lbl[:, 1:-1] + lbl[:, :-2]
                        temp_loss = nn.functional.mse_loss(pred_acc, tgt_acc)
                    else:
                        temp_loss = torch.tensor(0.0, device=device)

                    loss = pos_loss + args.temporal_weight * temp_loss

                val_losses.append(loss.item())
                val_pos_losses.append(pos_loss.item())
                val_pred_means.append(pred.mean().item())
                val_pred_stds.append(pred.std().item())

        avg_val = np.mean(val_losses)
        avg_val_pos = np.mean(val_pos_losses)
        avg_pred_mean = np.mean(val_pred_means)
        avg_pred_std = np.mean(val_pred_stds)

        epoch_time = time.time() - epoch_start

        log.info(
            "Epoch %3d/%d | train=%.6f val=%.6f pos=%.6f | "
            "pred_μ=%.3f pred_σ=%.3f | lr=%.2e | %.1fs",
            epoch, args.epochs, avg_train, avg_val, avg_val_pos,
            avg_pred_mean, avg_pred_std,
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        writer.add_scalar("val/loss", avg_val, epoch)
        writer.add_scalar("val/pos_loss", avg_val_pos, epoch)
        writer.add_scalar("val/pred_mean", avg_pred_mean, epoch)
        writer.add_scalar("val/pred_std", avg_pred_std, epoch)
        writer.add_scalar("train/loss_epoch", avg_train, epoch)
        writer.add_scalar("train/pos_loss_epoch", avg_train_pos, epoch)

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

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "model_config": model_config,
                "val_loss": avg_val,
                "global_step": global_step,
                "model_config": model_config,
            }, checkpoint_dir / f"tcn_epoch{epoch}.pt")

    writer.close()
    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    train()
