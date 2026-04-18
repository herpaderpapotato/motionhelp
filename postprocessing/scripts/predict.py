"""Apply the refinement model to postprocess TCN predictions.

Usage:
    # Refine predictions for a scene with pre-extracted features:
    python postprocessing/scripts/predict.py --input data/predictions/scene_00018_t00926_40s.npy

    # Refine predictions from a numpy array:
    python postprocessing/scripts/predict.py --input predictions.npy --output refined.npy

    # Refine and compare to labels:
    python postprocessing/scripts/predict.py --input predictions.npy --labels labels.npy --plot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from postprocessing.src.models.refinement import RefinementTCN


def load_refinement_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[RefinementTCN, dict]:
    """Load a refinement model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = RefinementTCN(
        channels=cfg["channels"],
        n_blocks=cfg["n_blocks"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg.get("dropout", 0.1),
        residual_mode=cfg.get("residual_mode", "logit"),
        delta_limit=cfg.get("delta_limit", 0.35),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"Loaded refinement model: epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.6f}")
    return model, cfg


def refine_predictions(
    model: RefinementTCN,
    predictions: np.ndarray,
    device: torch.device,
    batch_size: int = 1,
) -> np.ndarray:
    """Apply the refinement model to a prediction array.

    Args:
        predictions: [T] or [N, T] array of predictions in [0, 1]
        device: torch device
        batch_size: batch size for processing

    Returns:
        Refined predictions with same shape as input
    """
    single = predictions.ndim == 1
    if single:
        predictions = predictions[np.newaxis, :]

    refined_parts = []
    with torch.no_grad():
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i + batch_size]
            x = torch.from_numpy(batch).float().to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
            refined_parts.append(out.float().cpu().numpy())

    refined = np.concatenate(refined_parts, axis=0)

    if single:
        refined = refined[0]

    return refined


def main():
    parser = argparse.ArgumentParser(description="Refine TCN predictions")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to predictions .npy file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save refined predictions")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Path to labels .npy for comparison")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "postprocessing" / "data" / "checkpoints" / "best_refinement.pt")
    parser.add_argument("--plot", action="store_true",
                        help="Show a comparison plot")
    parser.add_argument("--save-plot", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, cfg = load_refinement_model(args.checkpoint, device)

    # Load predictions
    predictions = np.load(str(args.input))
    print(f"Input: {predictions.shape}, range=[{predictions.min():.4f}, {predictions.max():.4f}]")

    # Refine
    refined = refine_predictions(model, predictions, device)
    print(f"Refined: {refined.shape}, range=[{refined.min():.4f}, {refined.max():.4f}]")

    # Compare to labels if available
    labels = None
    if args.labels and args.labels.exists():
        labels = np.load(str(args.labels))
        n = min(len(predictions), len(labels), len(refined))
        predictions = predictions[:n]
        labels = labels[:n]
        refined = refined[:n]

        orig_mse = float(np.mean((predictions - labels) ** 2))
        refined_mse = float(np.mean((refined - labels) ** 2))
        orig_mae = float(np.mean(np.abs(predictions - labels)))
        refined_mae = float(np.mean(np.abs(refined - labels)))

        print(f"\n  Original:  MSE={orig_mse:.6f}  MAE={orig_mae:.6f}  RMSE={orig_mse**0.5:.4f}")
        print(f"  Refined:   MSE={refined_mse:.6f}  MAE={refined_mae:.6f}  RMSE={refined_mse**0.5:.4f}")
        improvement = (1 - refined_mse / orig_mse) * 100 if orig_mse > 0 else 0
        print(f"  Improvement: {improvement:+.1f}% MSE reduction")

    # Save output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.output), refined.astype(np.float32))
        print(f"\nSaved: {args.output}")

    # Plot
    if args.plot or args.save_plot:
        import matplotlib
        if args.save_plot and not args.plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fps = 30.0
        t = np.arange(len(refined)) / fps

        fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

        axes[0].plot(t, predictions, lw=1.2, alpha=0.7, color="darkorange", label="Original TCN")
        axes[0].plot(t, refined, lw=1.5, color="royalblue", label="Refined")
        if labels is not None:
            axes[0].plot(t, labels, lw=1.0, alpha=0.5, color="green", label="Ground truth")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("Position")
        axes[0].legend(fontsize=9)
        axes[0].set_title("Refinement comparison")
        axes[0].grid(alpha=0.3)

        if labels is not None:
            orig_err = predictions - labels
            refined_err = refined - labels
            axes[1].fill_between(t, orig_err, color="darkorange", alpha=0.3, label="Original error")
            axes[1].fill_between(t, refined_err, color="royalblue", alpha=0.3, label="Refined error")
            axes[1].axhline(0, color="black", lw=0.8)
            axes[1].set_ylim(-0.5, 0.5)
            axes[1].set_ylabel("Error")
            axes[1].set_xlabel("Time (s)")
            axes[1].legend(fontsize=9)
            axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if args.save_plot:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.save_plot, dpi=150)
            print(f"Saved plot: {args.save_plot}")
        if args.plot:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
