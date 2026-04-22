"""Evaluate scenes with a DispositionTCN checkpoint and store metrics in review.json.

Watches data/processed/ for scenes that have spatial features and labels, runs
DispositionTCN prediction, and stores regression metrics in each scene's
review.json.

Usage:
    python scripts/evaluate_scenes.py                    # evaluate once and exit
    python scripts/evaluate_scenes.py --watch            # poll every 60 s
    python scripts/evaluate_scenes.py --overwrite        # re-evaluate all scenes
    python scripts/evaluate_scenes.py --scenes scene_00018_t00926_40s
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.curation import discover_scenes, read_review, write_review
from src.data.spatial import legacy_conf_path, read_spatial_features_h5, spatial_feature_path
from src.models.dispositiontcn import DispositionTCN, extract_disposition_config
from src.training.funscript_metrics import compute_regression_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("data/models/checkpoints_disposition/best_disposition.pt")
DEFAULT_MODEL_NAME = "vrlens-finetunes-multiclass-v2-yolo26m-pose"


def load_model(checkpoint_path: Path, device: torch.device):
    """Load DispositionTCN model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model_type = cfg.get("model_type")
    if model_type not in (None, "disposition_tcn"):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a disposition model "
            f"(model_type={model_type!r})"
        )
    model = DispositionTCN(**extract_disposition_config(cfg))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    log.info(
        "Loaded checkpoint: epoch %s, val_loss=%s [disposition]",
        ckpt.get("epoch", "?"),
        f"{ckpt.get('val_loss'):.6f}" if isinstance(ckpt.get("val_loss"), (int, float)) else ckpt.get("val_loss"),
    )
    return model, cfg, ckpt.get("data_config", {})


def _split_predictions(pred: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if pred.ndim == 2:
        return pred, []
    if pred.ndim == 3:
        return pred[:, 0], [pred[:, idx] for idx in range(1, pred.shape[1])]
    raise ValueError(f"Unsupported prediction shape {tuple(pred.shape)}")


def sliding_window_predict(
    model,
    spatial: np.ndarray,
    conf: np.ndarray,
    device: torch.device,
    seq_len: int = 120,
    stride: int = 60,
) -> np.ndarray:
    """Slide a window over the full sequence and average overlapping predictions."""
    n_frames = len(spatial)
    pred_sum = np.zeros(n_frames, dtype=np.float32)
    pred_count = np.zeros(n_frames, dtype=np.float32)

    if n_frames < seq_len:
        sp = torch.from_numpy(spatial).float().unsqueeze(0).to(device)
        co = torch.from_numpy(conf).float().unsqueeze(0).to(device)
        pad = seq_len - n_frames
        sp = F.pad(sp, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad))
        co = F.pad(co, (0, 0, 0, pad))
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(sp, co)
        main_out, _ = _split_predictions(out)
        return main_out[0, :n_frames].float().cpu().numpy()

    starts = list(range(0, n_frames - seq_len + 1, stride))
    if starts[-1] + seq_len < n_frames:
        starts.append(n_frames - seq_len)

    with torch.no_grad():
        for start in starts:
            end = start + seq_len
            sp = torch.from_numpy(spatial[start:end]).float().unsqueeze(0).to(device)
            co = torch.from_numpy(conf[start:end]).float().unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(sp, co)

            main_out, _ = _split_predictions(out)
            p = main_out[0].float().cpu().numpy()
            weight = np.bartlett(seq_len).astype(np.float32) + 0.01
            pred_sum[start:end] += p * weight
            pred_count[start:end] += weight

    mask = pred_count > 0
    pred_sum[mask] /= pred_count[mask]
    return pred_sum


def evaluate_scene(
    scene_dir: Path,
    model,
    device: torch.device,
    spatial_model_name: str,
    seq_len: int = 120,
    stride: int = 60,
) -> dict[str, float] | None:
    """Run TCN prediction on a scene and return regression metrics. None on error."""
    scene_id = scene_dir.name
    spatial_path = spatial_feature_path(scene_dir, spatial_model_name)
    conf_path = legacy_conf_path(scene_dir, spatial_model_name)
    labels_path = scene_dir / "labels.npy"

    for p, name in [(spatial_path, "spatial"), (labels_path, "labels")]:
        if not p.exists():
            log.debug("SKIP %s — missing %s", scene_id, name)
            return None

    spatial, conf, _ = read_spatial_features_h5(
        spatial_path,
        legacy_conf_path=conf_path,
    )
    labels = np.load(str(labels_path)).astype(np.float32)

    # Align lengths
    n = min(len(spatial), len(conf), len(labels))
    spatial = spatial[:n]
    conf = conf[:n]
    labels = labels[:n]

    predictions = sliding_window_predict(model, spatial, conf, device, seq_len, stride)

    pred_tensor = torch.from_numpy(predictions).float().unsqueeze(0)
    label_tensor = torch.from_numpy(labels).float().unsqueeze(0)
    metrics = compute_regression_metrics(
        pred_tensor,
        label_tensor,
        spectral_kernel=15,
    )
    return {
        "mse": float(metrics["pos_mse"].item()),
        "event_mse": float(metrics["event_mse"].item()),
        "active_mse": float(metrics["active_mse"].item()),
        "vel_mae": float(metrics["vel_mae"].item()),
        "acc_mae": float(metrics["acc_mae"].item()),
        "spec_mse": float(metrics["spec_mse"].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scenes with DispositionTCN and store MSE")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--watch", action="store_true",
                        help="Keep running, re-scan on --poll-interval")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between scans in watch mode (default: 60)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-evaluate scenes that already have MSE")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Only evaluate these scene IDs (default: all ready)")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the spatial feature model stem used for scene spatial caches",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    # Load model
    model, model_cfg, data_cfg = load_model(args.checkpoint, device)
    spatial_model_name = Path(args.model_name or data_cfg.get("model_name") or DEFAULT_MODEL_NAME).stem
    log.info("Spatial model: %s", spatial_model_name)

    processed_dir = args.data_dir / "processed"

    def run_once() -> None:
        all_scenes = discover_scenes(args.data_dir, include_rejected=False, require_labels=True)

        pending = []
        for sid, state in all_scenes:
            if args.scenes and sid not in args.scenes:
                continue
            scene_dir = processed_dir / sid

            spatial_path = spatial_feature_path(scene_dir, spatial_model_name)
            if not spatial_path.exists():
                continue

            # Check if already evaluated
            if not args.overwrite:
                review = read_review(scene_dir)
                if review.get("mse") is not None and review["mse"] != "":
                    continue

            pending.append(sid)

        log.info("Found %d scenes to evaluate", len(pending))
        ok = failed = 0
        for i, sid in enumerate(pending):
            scene_dir = processed_dir / sid
            try:
                metrics = evaluate_scene(
                    scene_dir,
                    model,
                    device,
                    spatial_model_name,
                    args.seq_len,
                    args.stride,
                )
                if metrics is not None:
                    # Update review.json with scene-level regression metrics.
                    review = read_review(scene_dir)
                    for key, value in metrics.items():
                        review[key] = f"{value:.6f}"
                    write_review(scene_dir, review)
                    log.info(
                        "[%d/%d] %s  MSE=%.4f active=%.4f vel_mae=%.4f",
                        i + 1,
                        len(pending),
                        sid,
                        metrics["mse"],
                        metrics["active_mse"],
                        metrics["vel_mae"],
                    )
                    ok += 1
                else:
                    failed += 1
            except Exception:
                log.exception("[%d/%d] %s failed", i + 1, len(pending), sid)
                failed += 1

        log.info("Evaluation complete: %d ok, %d failed", ok, failed)

    if args.watch:
        log.info("Watch mode (poll every %ds) — Ctrl+C to stop", args.poll_interval)
        while True:
            run_once()
            time.sleep(args.poll_interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
