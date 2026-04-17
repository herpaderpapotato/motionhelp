"""Training script for the dense-flow lite funscript model.

Usage:
    python scripts/train_lite.py --epochs 30
    python scripts/train_lite.py --epochs 3 --max-train-scenes 128 --max-val-scenes 32

This experiment uses multiclass scene tensors, selects one primary partner plus
the fixed beholder slot, keeps the dense RAFT flow map, and trains a smaller
temporal model with a gated embedding residual.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lite_tcn import FunscriptLiteTCN
from src.training.funscript_metrics import compute_regression_metrics


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

log = logging.getLogger(__name__)


class LiteMotionDataset(Dataset):
    """Load multiclass keypoints, embeddings, dense flow, and labels for lite training."""

    KP_FILE = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
    EMB_FILE = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
    EMB_FILE_META = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.json"
    DENSE_FLOW_FILE = "flow/raft_dense_32x32_s0.5.npy"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        seq_len: int = 120,
        stride: int = 60,
        n_partners: int = 5,
        n_beholders: int = 1,
        n_keypoints: int = 21,
        embed_dim: int = 512,
        flow_dense_size: int = 32,
        max_scenes: int | None = None,
        max_sequences: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.n_partners = n_partners
        self.n_beholders = n_beholders
        self.n_total = n_partners + n_beholders
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dense_size = flow_dense_size
        self.stats_path: Path | None = None

        with open(self.data_dir / "splits" / f"{split}.json", encoding="utf-8") as handle:
            video_ids = json.load(handle)
        if max_scenes is not None:
            video_ids = video_ids[:max_scenes]

        self._load_stats()

        self.sequences: list[tuple[str, int]] = []
        processed = self.data_dir / "processed"
        skipped = 0

        for vid_id in video_ids:
            vid_dir = processed / vid_id
            kp_path = vid_dir / self.KP_FILE
            emb_path = vid_dir / self.EMB_FILE
            emb_meta_path = vid_dir / self.EMB_FILE_META
            flow_path = vid_dir / self.DENSE_FLOW_FILE
            label_path = vid_dir / "labels.npy"
            required = [kp_path, emb_path, emb_meta_path, flow_path, label_path]
            if not all(path.exists() for path in required):
                skipped += 1
                continue

            review_path = vid_dir / "review.json"
            if review_path.exists():
                review = json.loads(review_path.read_text(encoding="utf-8"))
                if review.get("status") == "rejected" or review.get("stage2_status") == "rejected":
                    skipped += 1
                    continue

            meta = json.loads(emb_meta_path.read_text(encoding="utf-8"))
            if meta.get("method") != "single_pass_hook_roi_align" or not meta.get("multiclass", False):
                skipped += 1
                continue

            n_frames = np.load(str(label_path), mmap_mode="r").shape[0]
            if n_frames < seq_len:
                skipped += 1
                continue

            for start in range(0, n_frames - seq_len + 1, stride):
                self.sequences.append((vid_id, start))
                if max_sequences is not None and len(self.sequences) >= max_sequences:
                    break
            if max_sequences is not None and len(self.sequences) >= max_sequences:
                break

        log.info(
            "Built %s lite dataset: %d sequences from %d scenes (%d skipped)",
            split,
            len(self.sequences),
            len(video_ids) - skipped,
            skipped,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        vid_dir = self.data_dir / "processed" / vid_id
        end = start + self.seq_len

        labels = np.load(str(vid_dir / "labels.npy"), mmap_mode="r")[start:end].copy().astype(np.float32)
        keypoints = np.load(str(vid_dir / self.KP_FILE), mmap_mode="r")[start:end].copy().astype(np.float32)
        embeddings = np.load(str(vid_dir / self.EMB_FILE), mmap_mode="r")[start:end].copy().astype(np.float32)
        flow = np.load(str(vid_dir / self.DENSE_FLOW_FILE), mmap_mode="r")[start:end].copy().astype(np.float32)

        if self.emb_mean is not None:
            embeddings = (embeddings - self.emb_mean) / (self.emb_std + 1e-8)
        if self.flow_dense_mean is not None:
            flow = (flow - self.flow_dense_mean) / (self.flow_dense_std + 1e-8)

        return {
            "keypoints": torch.from_numpy(keypoints).float(),
            "embeddings": torch.from_numpy(embeddings).float(),
            "flow": torch.from_numpy(flow).float(),
            "labels": torch.from_numpy(labels).float(),
        }

    def sample_high_variance(
        self,
        n: int = 8,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        pool = min(2000, len(self.sequences))
        indices = np.random.choice(len(self.sequences), pool, replace=False)
        stds: list[float] = []
        for sample_idx in indices:
            vid_id, start = self.sequences[sample_idx]
            label_path = self.data_dir / "processed" / vid_id / "labels.npy"
            label_slice = np.load(str(label_path), mmap_mode="r")[start:start + self.seq_len]
            stds.append(float(label_slice.std()))
        top_idx = np.array(stds).argsort()[-n:][::-1]
        batch = [self[indices[i]] for i in top_idx]
        out = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
        if device is not None:
            out = {key: value.to(device) for key, value in out.items()}
        return out

    def _load_stats(self) -> None:
        stats_path = self.data_dir / "featurestats" / "feature_stats.npz"
        if not stats_path.exists():
            log.warning("No featurestats/feature_stats.npz found; lite training will be unnormalized")
            self.emb_mean = None
            self.emb_std = None
            self.flow_dense_mean = None
            self.flow_dense_std = None
            return

        self.stats_path = stats_path
        stats = np.load(stats_path)

        emb_mean = stats.get("emb_mean")
        emb_std = stats.get("emb_std")
        expected = self.n_total * self.embed_dim
        if emb_mean is not None and emb_mean.shape[0] == expected:
            self.emb_mean = emb_mean.reshape(self.n_total, self.embed_dim)
            self.emb_std = emb_std.reshape(self.n_total, self.embed_dim)
        else:
            self.emb_mean = None
            self.emb_std = None
            log.warning("Embedding stats shape mismatch for lite model; embeddings will be unnormalized")

        flow_dense_mean = stats.get("flow_dense_mean")
        flow_dense_std = stats.get("flow_dense_std")
        if flow_dense_mean is not None and flow_dense_mean.shape == (2, 1, 1):
            self.flow_dense_mean = flow_dense_mean.astype(np.float32)
            self.flow_dense_std = flow_dense_std.astype(np.float32)
        else:
            self.flow_dense_mean = None
            self.flow_dense_std = None
            log.warning("Dense flow stats missing or mismatched; dense flow will be unnormalized")


def _compute_lite_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    main_pred = pred[:, 0]  # [B, T]
    metric_batch = compute_regression_metrics(
        main_pred,
        labels,
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
    return loss, main_pred, metric_batch


def _metric_means(metric_history: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in metric_history.items()
    }


def train() -> None:
    parser = argparse.ArgumentParser(description="Train lite dense-flow funscript model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--velocity-weight", type=float, default=0.05)
    parser.add_argument("--spectral-weight", type=float, default=0.05)
    parser.add_argument("--spectral-kernel", type=int, default=15)
    parser.add_argument("--event-weight", type=float, default=0.25)
    parser.add_argument("--event-activity-gain", type=float, default=3.0)
    parser.add_argument("--event-activity-power", type=float, default=1.0)
    parser.add_argument("--active-quantile", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--force-random-seed", action="store_true", help="Don't set random seeds; allow nondeterminism for potential speedup")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineWarmupLR",
        choices=["OneCycleLR", "CosineAnnealingLR", "CosineWarmupLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--n-partners", type=int, default=5)
    parser.add_argument("--n-beholders", type=int, default=1)
    parser.add_argument("--n-beholder-keypoints", type=int, default=7)
    parser.add_argument("--flow-dense-size", type=int, default=32)
    parser.add_argument(
        "--flow-flip",
        type=str,
        default="none",
        choices=sorted(FunscriptLiteTCN.VALID_FLOW_FLIPS),
        help="Apply a corrective flip to dense flow before global encoding and keypoint sampling.",
    )
    parser.add_argument("--checkpoint-name", type=str, default="best_lite_tcn.pt")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-val-scenes", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument("--plot-every", type=int, default=2)
    args = parser.parse_args()

    if not 0.0 <= args.event_weight <= 1.0:
        parser.error("--event-weight must be between 0 and 1")
    if not 0.0 < args.active_quantile <= 1.0:
        parser.error("--active-quantile must be in (0, 1]")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    if not args.force_random_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    train_ds = LiteMotionDataset(
        data_dir=args.data_dir,
        split="train",
        seq_len=args.seq_len,
        stride=args.stride,
        n_partners=args.n_partners,
        n_beholders=args.n_beholders,
        n_keypoints=21,
        embed_dim=512,
        flow_dense_size=args.flow_dense_size,
        max_scenes=args.max_train_scenes,
        max_sequences=args.max_train_sequences,
    )
    val_ds = LiteMotionDataset(
        data_dir=args.data_dir,
        split="val",
        seq_len=args.seq_len,
        stride=args.stride,
        n_partners=args.n_partners,
        n_beholders=args.n_beholders,
        n_keypoints=21,
        embed_dim=512,
        flow_dense_size=args.flow_dense_size,
        max_scenes=args.max_val_scenes,
        max_sequences=args.max_val_sequences,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Lite training dataset is empty; check multiclass features, dense flow, and split limits")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    log.info("Train: %d sequences (%d batches)", len(train_ds), len(train_loader))
    log.info("Val:   %d sequences (%d batches)", len(val_ds), len(val_loader))

    model = FunscriptLiteTCN(
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        n_partners=args.n_partners,
        n_beholders=args.n_beholders,
        n_keypoints=21,
        n_beholder_keypoints=args.n_beholder_keypoints,
        embed_dim=512,
        flow_mode="dense",
        flow_dense_size=args.flow_dense_size,
        flow_flip=args.flow_flip,
    ).to(device)
    params = model.count_parameters()
    log.info("Model: %s trainable / %s total parameters", f"{params['trainable']:,}", f"{params['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        restart_period = max(args.epochs * len(train_loader) // 4, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_period,
            T_mult=1,
            eta_min=args.lr * 0.01,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=args.lr * 0.001,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_name = f"lite_tcn_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    checkpoint_dir = Path("data/models/checkpoints_tcn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    model_config = {
        "model_type": "lite_tcn",
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "dropout": args.dropout,
        "n_partners": args.n_partners,
        "n_beholders": args.n_beholders,
        "n_keypoints": 21,
        "n_beholder_keypoints": args.n_beholder_keypoints,
        "embed_dim": 512,
        "flow_mode": "dense",
        "flow_dense_size": args.flow_dense_size,
        "flow_flip": args.flow_flip,
    }
    data_config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "stats_path": str(train_ds.stats_path) if train_ds.stats_path is not None else None,
        "max_train_scenes": args.max_train_scenes,
        "max_val_scenes": args.max_val_scenes,
        "max_train_sequences": args.max_train_sequences,
        "max_val_sequences": args.max_val_sequences,
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

    best_val_loss = float("inf")
    early_stop_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses: list[float] = []
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
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            embeddings = batch["embeddings"].to(device, non_blocking=True)
            flow = batch["flow"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(keypoints, embeddings, flow)
                loss, main_pred, metric_batch = _compute_lite_loss(pred, labels, args)

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
                writer.add_scalar("train/pos_mse", metric_batch["pos_mse"].item(), global_step)
                writer.add_scalar("train/event_mse", metric_batch["event_mse"].item(), global_step)
                writer.add_scalar("train/active_mse", metric_batch["active_mse"].item(), global_step)
                writer.add_scalar("train/vel_mae", metric_batch["vel_mae"].item(), global_step)
                writer.add_scalar("train/acc_mae", metric_batch["acc_mae"].item(), global_step)
                writer.add_scalar("train/pred_mean", main_pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", main_pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

        avg_train = float(np.mean(train_losses))
        avg_train_metrics = _metric_means(train_metric_history)

        model.eval()
        val_losses: list[float] = []
        val_pred_means: list[float] = []
        val_pred_stds: list[float] = []
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
                keypoints = batch["keypoints"].to(device, non_blocking=True)
                embeddings = batch["embeddings"].to(device, non_blocking=True)
                flow = batch["flow"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(keypoints, embeddings, flow)
                    loss, main_pred, metric_batch = _compute_lite_loss(pred, labels, args)

                val_losses.append(loss.item())
                for key in val_metric_history:
                    val_metric_history[key].append(metric_batch[key].item())
                val_pred_means.append(main_pred.mean().item())
                val_pred_stds.append(main_pred.std().item())

        avg_val = float(np.mean(val_losses))
        avg_val_metrics = _metric_means(val_metric_history)
        avg_pred_mean = float(np.mean(val_pred_means))
        avg_pred_std = float(np.mean(val_pred_stds))
        epoch_time = time.time() - epoch_start

        log.info(
            "Epoch %3d/%d | train=%.6f val=%.6f pos=%.6f event=%.6f active=%.6f vel_mae=%.6f | pred_mu=%.3f pred_sigma=%.3f | lr=%.2e | %.1fs",
            epoch,
            args.epochs,
            avg_train,
            avg_val,
            avg_val_metrics["pos_mse"],
            avg_val_metrics["event_mse"],
            avg_val_metrics["active_mse"],
            avg_val_metrics["vel_mae"],
            avg_pred_mean,
            avg_pred_std,
            optimizer.param_groups[0]["lr"],
            epoch_time,
        )

        writer.add_scalar("train/loss_epoch", avg_train, epoch)
        writer.add_scalar("train/pos_mse_epoch", avg_train_metrics["pos_mse"], epoch)
        writer.add_scalar("train/event_mse_epoch", avg_train_metrics["event_mse"], epoch)
        writer.add_scalar("val/loss", avg_val, epoch)
        writer.add_scalar("val/pos_mse", avg_val_metrics["pos_mse"], epoch)
        writer.add_scalar("val/event_mse", avg_val_metrics["event_mse"], epoch)
        writer.add_scalar("val/active_mse", avg_val_metrics["active_mse"], epoch)
        writer.add_scalar("val/vel_mae", avg_val_metrics["vel_mae"], epoch)
        writer.add_scalar("val/acc_mae", avg_val_metrics["acc_mae"], epoch)
        writer.add_scalar("val/pred_mean", avg_pred_mean, epoch)
        writer.add_scalar("val/pred_std", avg_pred_std, epoch)

        if args.scheduler == "ReduceLROnPlateau":
            scheduler.step(avg_val)

        if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == 1):
            with torch.no_grad():
                samples = val_ds.sample_high_variance(n=min(8, len(val_ds)), device=device)
                pred_s = model(samples["keypoints"], samples["embeddings"], samples["flow"])[:, 0]
                pred_np = pred_s.float().cpu().numpy()
                labels_np = samples["labels"].float().cpu().numpy()

            n_plots = min(8, len(pred_np))
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=False)
            if n_plots == 1:
                axes = [axes]
            for plot_idx, ax in enumerate(axes):
                ax.plot(labels_np[plot_idx], label="target", alpha=0.85, lw=1.5, color="steelblue")
                ax.plot(pred_np[plot_idx], label="lite", alpha=0.9, lw=1.5, color="darkorange")
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(f"#{plot_idx}", fontsize=7)
                if plot_idx == 0:
                    ax.legend(fontsize=7)
                    ax.set_title(f"Epoch {epoch} lite predictions")
            fig.tight_layout()
            writer.add_figure("Predictions/lite_overlay", fig, epoch)
            plt.close(fig)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val,
                    "global_step": global_step,
                    "model_config": model_config,
                    "data_config": data_config,
                    "metric_config": metric_config,
                },
                checkpoint_path,
            )
            log.info("  -> New best val loss: %.6f (%s)", avg_val, checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stopping_patience:
                log.info(
                    "Early stopping after %d unimproved epochs (best val=%.6f)",
                    early_stop_counter,
                    best_val_loss,
                )
                break

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val,
                    "global_step": global_step,
                    "model_config": model_config,
                    "data_config": data_config,
                    "metric_config": metric_config,
                },
                checkpoint_dir / f"lite_tcn_epoch{epoch}.pt",
            )

    writer.close()
    log.info("Lite training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    train()