"""Train DispositionTCN on spatial RoI features.

Usage:
    python scripts/train_disposition.py --epochs 30 --batch-size 4
    python scripts/train_disposition.py --epochs 5 --batch-size 2  # quick test

Loads spatial features extracted by scripts/extract_spatial.py and trains
a DispositionTCN for per-frame position prediction.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.dispositiontcn import DispositionTCN, extract_disposition_config
from src.training.funscript_metrics import compute_regression_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "vrlens-finetunes-multiclass-v2-yolo26m-pose"


class SpatialDataset(Dataset):
    """Loads spatial RoI features + labels for DispositionTCN training.

    Expects:
        data/processed/{scene_id}/spatial/{model_name}.h5        [T, N, C, H, W] float16
        data/processed/{scene_id}/spatial/{model_name}_conf.h5   [T, N] float32
        data/processed/{scene_id}/labels.npy                     [T] float32
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        seq_len: int = 120,
        stride: int = 60,
        model_name: str = DEFAULT_MODEL_NAME,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.model_name = model_name
        self.augment = augment

        split_file = self.data_dir / "splits" / f"disposition_{split}.json"
        with open(split_file) as f:
            video_ids = json.load(f)

        self.sequences: list[tuple[str, int]] = []
        processed = self.data_dir / "processed"
        skipped = 0

        for vid_id in video_ids:
            vid_dir = processed / vid_id
            spatial_path = vid_dir / "spatial" / f"{self.model_name}.h5"
            conf_path = vid_dir / "spatial" / f"{self.model_name}_conf.h5"
            label_path = vid_dir / "labels.npy"

            if not all(p.exists() for p in [spatial_path, conf_path, label_path]):
                skipped += 1
                continue

            n_frames = np.load(str(label_path), mmap_mode="r").shape[0]
            if n_frames < seq_len:
                skipped += 1
                continue

            for start in range(0, n_frames - seq_len + 1, stride):
                self.sequences.append((vid_id, start))

        log.info(
            "SpatialDataset [%s]: %d sequences from %d videos (%d skipped)",
            split, len(self.sequences), len(video_ids) - skipped, skipped,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        vid_dir = self.data_dir / "processed" / vid_id
        end = start + self.seq_len

        with h5py.File(str(vid_dir / "spatial" / f"{self.model_name}.h5"), "r") as f:
            spatial = f["spatial"][start:end].copy()
        with h5py.File(str(vid_dir / "spatial" / f"{self.model_name}_conf.h5"), "r") as f:
            conf = f["conf"][start:end].copy()
        labels = np.load(str(vid_dir / "labels.npy"), mmap_mode="r")[start:end].copy()

        spatial = torch.from_numpy(spatial).float()   # [T, N, C, H, W]
        conf = torch.from_numpy(conf).float()          # [T, N]
        labels = torch.from_numpy(labels).float()       # [T]

        if self.augment:
            # Time reversal
            if torch.rand(1).item() < 0.3:
                spatial = spatial.flip(0)
                conf = conf.flip(0)
                labels = labels.flip(0)
            # Position inversion
            if torch.rand(1).item() < 0.3:
                labels = 1.0 - labels

        return {"spatial": spatial, "conf": conf, "labels": labels}

    def sample_random(self, n: int, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        idx = np.random.choice(len(self.sequences), min(n, len(self.sequences)), replace=False)
        batch = [self[i] for i in idx]
        out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out


def train() -> None:
    parser = argparse.ArgumentParser(description="Train DispositionTCN")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--n-persons", type=int, default=1)
    parser.add_argument("--use-ddl", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Datasets
    train_ds = SpatialDataset(
        args.data_dir, "train", args.seq_len, args.stride,
        model_name=args.model_name, augment=True,
    )
    val_ds = SpatialDataset(
        args.data_dir, "val", args.seq_len, args.stride,
        model_name=args.model_name, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=True,
    )

    log.info("Train: %d sequences (%d batches)", len(train_ds), len(train_loader))
    log.info("Val:   %d sequences (%d batches)", len(val_ds), len(val_loader))

    # Infer in_channels from first sample
    sample = train_ds[0]
    in_channels = sample["spatial"].shape[2]  # C dimension of [T, N, C, H, W]
    log.info("Inferred in_channels=%d from data", in_channels)

    # Model
    model = DispositionTCN(
        in_channels=in_channels,
        roi_size=args.roi_size,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        n_persons=args.n_persons,
        encoder_dim=args.encoder_dim,
        use_ddl=args.use_ddl,
    ).to(device)

    params = model.count_parameters()
    log.info("Model: %s trainable / %s total parameters",
             f"{params['trainable']:,}", f"{params['total']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=max(1, args.epochs * len(train_loader) // 4),
    #     T_mult=1,
    #     eta_min=args.lr * 0.01,
    # )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Logging
    run_name = f"disposition_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    checkpoint_dir = Path("data/models/checkpoints_disposition")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "model_type": "disposition_tcn",
        "in_channels": in_channels,
        "roi_size": args.roi_size,
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "n_persons": args.n_persons,
        "encoder_dim": args.encoder_dim,
        "use_ddl": args.use_ddl,
    }
    data_config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "model_name": args.model_name,
        "roi_size": args.roi_size,
    }

    # Training loop
    best_val_loss = float("inf")
    early_stop_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_losses = []
        train_metrics = {"pos_mse": [], "vel_mse": [], "acc_mse": []}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
            spatial = batch["spatial"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(spatial, conf)  # [B, T]
                metrics = compute_regression_metrics(
                    pred, lbl, spectral_kernel=15,
                )
                loss = (metrics["pos_mse"]
                        + args.velocity_weight * metrics["vel_mse"]
                        + args.temporal_weight * metrics["acc_mse"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            for key in train_metrics:
                train_metrics[key].append(metrics[key].item())
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/pos_mse", metrics["pos_mse"].item(), global_step)
                writer.add_scalar("train/vel_mse", metrics["vel_mse"].item(), global_step)
                writer.add_scalar("train/pred_mean", pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

        avg_train = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        val_metrics = {"pos_mse": [], "vel_mse": [], "acc_mse": [], "vel_mae": []}

        with torch.no_grad():
            for batch in val_loader:
                spatial = batch["spatial"].to(device, non_blocking=True)
                conf = batch["conf"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(spatial, conf)
                    metrics = compute_regression_metrics(pred, lbl, spectral_kernel=15)
                    loss = (metrics["pos_mse"]
                            + args.velocity_weight * metrics["vel_mse"]
                            + args.temporal_weight * metrics["acc_mse"])

                val_losses.append(loss.item())
                for key in val_metrics:
                    val_metrics[key].append(metrics[key].item())

        avg_val = np.mean(val_losses)
        avg_val_metrics = {k: float(np.mean(v)) for k, v in val_metrics.items()}
        epoch_time = time.time() - epoch_start

        log.info(
            "Epoch %3d/%d | train=%.6f val=%.6f pos=%.6f vel_mae=%.6f | lr=%.2e | %.1fs",
            epoch, args.epochs, avg_train, avg_val,
            avg_val_metrics["pos_mse"], avg_val_metrics["vel_mae"],
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        writer.add_scalar("val/loss", avg_val, epoch)
        writer.add_scalar("val/pos_mse", avg_val_metrics["pos_mse"], epoch)
        writer.add_scalar("val/vel_mse", avg_val_metrics["vel_mse"], epoch)
        writer.add_scalar("train/loss_epoch", avg_train, epoch)

        # Prediction plots every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if epoch == 1:
                    plot_samples = val_ds.sample_random(n=4, device=device)
                sp = plot_samples["spatial"]
                co = plot_samples["conf"]
                lb = plot_samples["labels"]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred_s = model(sp, co)

                pred_s = pred_s.float().cpu().numpy()
                lb_np = lb.float().cpu().numpy()

            n_plots = min(4, len(pred_s))
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots))
            if n_plots == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                ax.plot(lb_np[i], label="target", alpha=0.85, lw=1.5, color="steelblue")
                ax.plot(pred_s[i], label="pred", alpha=0.85, lw=1.5, color="darkorange")
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(f"#{i}", fontsize=7)
                if i == 0:
                    ax.legend(fontsize=7)
                    ax.set_title(f"Epoch {epoch} predictions")
            fig.tight_layout()
            writer.add_figure("Predictions/overlay", fig, epoch)
            plt.close(fig)

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "global_step": global_step,
                "model_config": model_config,
                "data_config": data_config,
            }, checkpoint_dir / "best_disposition.pt")
            log.info("  -> New best val loss: %.6f", avg_val)
        else:
            early_stop_counter += 1
            if args.early_stopping_patience > 0 and early_stop_counter >= args.early_stopping_patience:
                log.info("Early stopping after %d epochs without improvement", early_stop_counter)
                break

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "global_step": global_step,
                "model_config": model_config,
                "data_config": data_config,
            }, checkpoint_dir / f"disposition_epoch{epoch}.pt")

    writer.close()
    log.info("Training complete. Best val loss: %.6f", best_val_loss)
    log.info("Run dir: %s", run_dir)


if __name__ == "__main__":
    train()
