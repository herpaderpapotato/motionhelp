"""Train the postprocessing refinement model.

Usage:
    python postprocessing/scripts/train.py
    python postprocessing/scripts/train.py --epochs 500 --channels 64 --n-blocks 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from postprocessing.src.models.refinement import RefinementTCN
from postprocessing.src.data.dataset import RefinementDataset


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalise differences in frame-to-frame velocity between pred and target."""
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    return nn.functional.mse_loss(pred_vel, target_vel)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 128) -> torch.Tensor:
    """Penalise differences in frequency content."""
    # Use a simple windowed FFT comparison
    pred_spec = torch.fft.rfft(pred, n=n_fft, dim=-1).abs()
    target_spec = torch.fft.rfft(target, n=n_fft, dim=-1).abs()
    return nn.functional.mse_loss(pred_spec, target_spec)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    temporal_weight: float = 0.001,
    spectral_weight: float = 0.001,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_tc = 0.0
    total_spec = 0.0
    n = 0

    for batch in loader:
        preds = batch["predictions"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            refined = model(preds)
            mse = nn.functional.mse_loss(refined, labels)
            tc = temporal_consistency_loss(refined, labels)
            spec = spectral_loss(refined, labels)
            loss = mse + temporal_weight * tc + spectral_weight * spec

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = preds.size(0)
        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_tc += tc.item() * bs
        total_spec += spec.item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "temporal": total_tc / n,
        "spectral": total_spec / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temporal_weight: float = 0.001,
    spectral_weight: float = 0.001,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_tc = 0.0
    total_spec = 0.0
    # Also track baseline (unrefined) loss
    total_baseline_mse = 0.0
    n = 0

    for batch in loader:
        preds = batch["predictions"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            refined = model(preds)
            mse = nn.functional.mse_loss(refined, labels)
            mae = nn.functional.l1_loss(refined, labels)
            tc = temporal_consistency_loss(refined, labels)
            spec = spectral_loss(refined, labels)
            loss = mse + temporal_weight * tc + spectral_weight * spec
            baseline_mse = nn.functional.mse_loss(preds, labels)

        bs = preds.size(0)
        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs
        total_tc += tc.item() * bs
        total_spec += spec.item() * bs
        total_baseline_mse += baseline_mse.item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "mae": total_mae / n,
        "temporal": total_tc / n,
        "spectral": total_spec / n,
        "baseline_mse": total_baseline_mse / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train refinement model")
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "postprocessing" / "data" / "prepared")
    parser.add_argument("--splits-dir", type=Path,
                        default=ROOT / "postprocessing" / "data" / "splits")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "postprocessing" / "data" / "checkpoints")
    parser.add_argument("--log-dir", type=Path,
                        default=ROOT / "postprocessing" / "runs")

    # Model
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temporal-weight", type=float, default=0.001)
    parser.add_argument("--spectral-weight", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. 'cpu' or 'cuda:0')")

    # Data
    parser.add_argument("--seq-len", type=int, default=1200,
                        help="Subsequence length for training (default: full 1200)")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    print("Loading datasets...")
    train_ds = RefinementDataset(
        args.prepared_dir, args.splits_dir, split="train",
        seq_len=args.seq_len, stride=args.stride, augment=True,
    )
    val_ds = RefinementDataset(
        args.prepared_dir, args.splits_dir, split="val",
        seq_len=args.seq_len, stride=args.stride, augment=False,
    )
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Val:   {len(val_ds)} sequences")

    if len(train_ds) == 0:
        print("ERROR: No training data found. Run prepare_data.py first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model
    model = RefinementTCN(
        channels=args.channels,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # Compute receptive field
    rf = 1
    for i in range(args.n_blocks):
        rf += 2 * (args.kernel_size - 1) * (2 ** i)
    print(f"Receptive field: {rf} frames ({rf / 30:.1f}s)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # Tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"refinement_{int(time.time())}"
        writer = SummaryWriter(str(args.log_dir / run_name))
    except ImportError:
        writer = None

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} {'Train':>10} {'Val':>10} {'Val MAE':>10} {'Baseline':>10} {'LR':>10}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            args.temporal_weight, args.spectral_weight,
        )
        val_metrics = validate(
            model, val_loader, device,
            args.temporal_weight, args.spectral_weight,
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        print(
            f"{epoch:5d} {train_metrics['loss']:10.6f} {val_metrics['loss']:10.6f} "
            f"{val_metrics['mae']:10.6f} {val_metrics['baseline_mse']:10.6f} {lr:10.2e}  "
            f"({elapsed:.1f}s)"
        )

        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", lr, epoch)

        # Checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "model_config": {
                    "channels": args.channels,
                    "n_blocks": args.n_blocks,
                    "kernel_size": args.kernel_size,
                    "dropout": args.dropout,
                },
                "training_config": {
                    "seq_len": args.seq_len,
                    "lr": args.lr,
                    "temporal_weight": args.temporal_weight,
                    "spectral_weight": args.spectral_weight,
                },
            }
            torch.save(ckpt, args.checkpoint_dir / "best_refinement.pt")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stopping_patience} epochs)")
            break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved: {args.checkpoint_dir / 'best_refinement.pt'}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
