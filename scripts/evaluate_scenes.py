"""Evaluate scenes with the TCN model and store MSE in review.json.

Watches data/processed/ for scenes that have keypoints, embeddings, flow, and
labels, runs TCN prediction, and stores the MSE in each scene's review.json.

Usage:
    python scripts/evaluate_scenes.py                    # evaluate once and exit
    python scripts/evaluate_scenes.py --watch            # poll every 60 s
    python scripts/evaluate_scenes.py --overwrite        # re-evaluate all scenes
    python scripts/evaluate_scenes.py --scenes scene_00018_t00926_40s
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.curation import discover_scenes, read_review, write_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

# Feature file names (must match predict_tcn.py / reprocess_embeddings.py)
KP_FILE = "keypoints/pose-vrlens-finetunes-large.npy"
EMB_FILE = "embeddings/pose-vrlens-finetunes-large.npy"
KP_FILE_MULTICLASS = "keypoints/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
EMB_FILE_MULTICLASS = "embeddings/vrlens-finetunes-multiclass-v2-yolo11m-pose.npy"
FLOW_FILE = "flow/raft_f64_s0.5.npy"


def load_model(checkpoint_path: Path, device: torch.device):
    """Load TCN model from checkpoint. Returns (model, model_config)."""
    from src.models.tcn import FunscriptTCN

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = FunscriptTCN(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    is_mc = cfg.get("n_partners") is not None
    log.info("Loaded checkpoint: epoch %s, val_loss=%.6f [%s]",
             ckpt.get("epoch", "?"), ckpt.get("val_loss", 0),
             "multiclass" if is_mc else "single-class")
    return model, cfg


def load_stats(data_dir: Path, n_persons: int = 10, embed_dim: int = 512, flow_dim: int = 64):
    """Load feature normalization stats."""
    stats_path = data_dir / "feature_stats.npz"
    if not stats_path.exists():
        # Try featurestats subdirectory
        alt = data_dir / "featurestats" / "feature_stats.npz"
        if alt.exists():
            stats_path = alt
        else:
            log.warning("No feature_stats.npz found — using un-normalized features")
            return None, None, None, None

    stats = np.load(stats_path)
    emb_mean = emb_std = None
    expected = n_persons * embed_dim
    if "emb_mean" in stats and stats["emb_mean"].shape[0] == expected:
        emb_mean = stats["emb_mean"].reshape(n_persons, embed_dim)
        emb_std = stats["emb_std"].reshape(n_persons, embed_dim)
    flow_mean = flow_std = None
    if "flow_mean" in stats and stats["flow_mean"].shape[0] == flow_dim:
        flow_mean = stats["flow_mean"]
        flow_std = stats["flow_std"]
    return emb_mean, emb_std, flow_mean, flow_std


def sliding_window_predict(
    model,
    keypoints: np.ndarray,
    embeddings: np.ndarray,
    flow: np.ndarray,
    device: torch.device,
    seq_len: int = 120,
    stride: int = 60,
) -> np.ndarray:
    """Slide a window over the full sequence and average overlapping predictions."""
    n_frames = len(keypoints)
    pred_sum = np.zeros(n_frames, dtype=np.float32)
    pred_count = np.zeros(n_frames, dtype=np.float32)

    if n_frames < seq_len:
        kp = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)
        emb = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
        fl = torch.from_numpy(flow).float().unsqueeze(0).to(device)
        pad = seq_len - n_frames
        kp = torch.nn.functional.pad(kp, (0, 0, 0, 0, 0, 0, 0, pad))
        emb = torch.nn.functional.pad(emb, (0, 0, 0, 0, 0, pad))
        fl = torch.nn.functional.pad(fl, (0, 0, 0, pad))
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(kp, emb, fl)
        return out[0, :n_frames].float().cpu().numpy()

    starts = list(range(0, n_frames - seq_len + 1, stride))
    if starts[-1] + seq_len < n_frames:
        starts.append(n_frames - seq_len)

    with torch.no_grad():
        for start in starts:
            end = start + seq_len
            kp = torch.from_numpy(keypoints[start:end]).float().unsqueeze(0).to(device)
            emb = torch.from_numpy(embeddings[start:end]).float().unsqueeze(0).to(device)
            fl = torch.from_numpy(flow[start:end]).float().unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(kp, emb, fl)

            p = out[0].float().cpu().numpy()
            weight = np.bartlett(seq_len).astype(np.float32) + 0.01
            pred_sum[start:end] += p * weight
            pred_count[start:end] += weight

    mask = pred_count > 0
    pred_sum[mask] /= pred_count[mask]
    return pred_sum


def evaluate_scene(
    scene_dir: Path,
    model,
    model_cfg: dict,
    device: torch.device,
    emb_mean: np.ndarray | None,
    emb_std: np.ndarray | None,
    flow_mean: np.ndarray | None,
    flow_std: np.ndarray | None,
    is_multiclass: bool,
    seq_len: int = 120,
    stride: int = 60,
) -> float | None:
    """Run TCN prediction on a scene and return MSE vs labels. None on error."""
    scene_id = scene_dir.name

    # Select feature files
    kp_file = KP_FILE_MULTICLASS if is_multiclass else KP_FILE
    emb_file = EMB_FILE_MULTICLASS if is_multiclass else EMB_FILE

    kp_path = scene_dir / kp_file
    emb_path = scene_dir / emb_file
    flow_path = scene_dir / FLOW_FILE
    labels_path = scene_dir / "labels.npy"

    for p, name in [(kp_path, "keypoints"), (emb_path, "embeddings"),
                    (flow_path, "flow"), (labels_path, "labels")]:
        if not p.exists():
            log.debug("SKIP %s — missing %s", scene_id, name)
            return None

    keypoints = np.load(str(kp_path))
    embeddings = np.load(str(emb_path))
    flow = np.load(str(flow_path))
    labels = np.load(str(labels_path))

    # Align lengths
    n = min(len(keypoints), len(embeddings), len(flow), len(labels))
    keypoints = keypoints[:n]
    embeddings = embeddings[:n]
    flow = flow[:n]
    labels = labels[:n]

    # Normalize
    if emb_mean is not None:
        embeddings = (embeddings - emb_mean) / (emb_std + 1e-8)
    if flow_mean is not None:
        flow = (flow - flow_mean) / (flow_std + 1e-8)

    predictions = sliding_window_predict(model, keypoints, embeddings, flow,
                                         device, seq_len, stride)

    mse = float(np.mean((predictions - labels) ** 2))
    return mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scenes with TCN and store MSE")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/models/checkpoints_tcn/best_tcn.pt"))
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
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    # Load model
    model, model_cfg = load_model(args.checkpoint, device)
    is_multiclass = model_cfg.get("n_partners") is not None
    n_total = (
        (model_cfg.get("n_partners", 5) + model_cfg.get("n_beholders", 1))
        if is_multiclass else model_cfg.get("n_persons", 10)
    )

    emb_mean, emb_std, flow_mean, flow_std = load_stats(args.data_dir, n_persons=n_total)

    processed_dir = args.data_dir / "processed"

    def run_once() -> None:
        all_scenes = discover_scenes(args.data_dir, include_rejected=False, require_labels=True)

        pending = []
        for sid, state in all_scenes:
            if args.scenes and sid not in args.scenes:
                continue
            scene_dir = processed_dir / sid

            # Check if scene has the required feature files
            kp_file = KP_FILE_MULTICLASS if is_multiclass else KP_FILE
            emb_file = EMB_FILE_MULTICLASS if is_multiclass else EMB_FILE
            if not all((scene_dir / f).exists() for f in [kp_file, emb_file, FLOW_FILE]):
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
                mse = evaluate_scene(
                    scene_dir, model, model_cfg, device,
                    emb_mean, emb_std, flow_mean, flow_std,
                    is_multiclass, args.seq_len, args.stride,
                )
                if mse is not None:
                    # Update review.json with MSE
                    review = read_review(scene_dir)
                    review["mse"] = f"{mse:.4f}"
                    write_review(scene_dir, review)
                    log.info("[%d/%d] %s  MSE=%.4f", i + 1, len(pending), sid, mse)
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
