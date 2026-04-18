"""Generate TCN predictions for all scenes in postprocessing splits.

For each scene in train/val, runs the existing TCN model with sliding-window
inference and saves the predictions alongside the labels.

Usage:
    python postprocessing/scripts/prepare_data.py
    python postprocessing/scripts/prepare_data.py --checkpoint data/models/checkpoints_tcn/best_tcn.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predict_tcn import (
    load_model,
    load_stats,
    sliding_window_predict,
    _extract_prediction_tensor,
)
from src.data.curation import embeddings_path, keypoints_path

DENSE_FLOW_FILE = "flow/raft_dense_32x32_s0.5.npy"
FLOW_FILE = "flow/raft_f64_s0.5.npy"


def prepare_scene(
    scene_id: str,
    model: torch.nn.Module,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    feature_model_name: str,
    flow_mode: str,
    seq_len: int,
    stride: int,
    emb_mean: np.ndarray | None,
    emb_std: np.ndarray | None,
    flow_mean: np.ndarray | None,
    flow_std: np.ndarray | None,
) -> bool:
    """Run TCN prediction for a single scene and save results."""
    scene_dir = data_dir / "processed" / scene_id
    out_dir = output_dir / scene_id

    if not scene_dir.exists():
        print(f"  SKIP {scene_id}: scene dir not found")
        return False
    
    review_file = scene_dir / "review.json"
    #### MODIFIED TO ALIGN WITH CURATION
    if review_file.exists():
        with open(review_file) as f:
            review = json.load(f)
        if review.get("status") != "approved":
            return False
    else:
        return False

    # Check if already prepared
    pred_path = out_dir / "predictions.npy"
    label_path = out_dir / "labels.npy"
    if pred_path.exists() and label_path.exists():
        return True

    # Load labels
    labels_file = scene_dir / "labels.npy"
    if not labels_file.exists():
        print(f"  SKIP {scene_id}: no labels")
        return False
    labels = np.load(str(labels_file))

    # Load features
    kp_path = keypoints_path(scene_dir, feature_model_name)
    emb_path = embeddings_path(scene_dir, feature_model_name)
    flow_path = scene_dir / (DENSE_FLOW_FILE if flow_mode == "dense" else FLOW_FILE)

    for p in [kp_path, emb_path, flow_path]:
        if not p.exists():
            print(f"  SKIP {scene_id}: missing {p.name}")
            return False

    keypoints = np.load(str(kp_path))
    embeddings = np.load(str(emb_path))
    flow = np.load(str(flow_path))
    if flow_mode == "dense":
        flow = flow.astype(np.float32)

    # Align frame counts
    n = min(len(keypoints), len(embeddings), len(flow), len(labels))
    keypoints = keypoints[:n]
    embeddings = embeddings[:n]
    flow = flow[:n]
    labels = labels[:n]

    # Normalize features
    if emb_mean is not None:
        embeddings = (embeddings - emb_mean) / (emb_std + 1e-8)
    if flow_mean is not None:
        flow = (flow - flow_mean) / (flow_std + 1e-8)

    # Run sliding window prediction
    predictions = sliding_window_predict(
        model, keypoints, embeddings, flow, device, seq_len, stride
    )

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(pred_path), predictions.astype(np.float32))
    np.save(str(label_path), labels.astype(np.float32))

    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare postprocessing data")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data" / "models" / "checkpoints_tcn" / "best_tcn.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "postprocessing" / "data" / "prepared")
    parser.add_argument("--splits-dir", type=Path,
                        default=ROOT / "postprocessing" / "data" / "splits")
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--feature-model-name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, model_cfg, data_cfg = load_model(args.checkpoint, device)

    feature_model_name = args.feature_model_name or data_cfg.get(
        "feature_model_name", "vrlens-finetunes-multiclass-v2-yolo26m-pose"
    )
    flow_mode = model_cfg.get("flow_mode", "summary")
    n_partners = model_cfg.get("n_partners", 5)
    n_beholders = model_cfg.get("n_beholders", 1)
    n_total = n_partners + n_beholders
    embed_dim = int(model_cfg.get("embed_dim", 512))

    # Load normalization stats
    emb_mean, emb_std, flow_mean, flow_std = load_stats(
        args.data_dir,
        n_persons=n_total,
        embed_dim=embed_dim,
        flow_mode=flow_mode,
        feature_model_name=feature_model_name,
    )

    # Process both splits
    for split in ["train", "val"]:
        split_file = args.splits_dir / f"{split}.json"
        if not split_file.exists():
            print(f"Split file not found: {split_file}")
            continue

        with open(split_file) as f:
            scene_ids = json.load(f)

        print(f"\nProcessing {split} split: {len(scene_ids)} scenes")
        success = 0
        for scene_id in tqdm(scene_ids, desc=split):
            ok = prepare_scene(
                scene_id, model, args.data_dir, args.output_dir, device,
                feature_model_name, flow_mode, args.seq_len, args.stride,
                emb_mean, emb_std, flow_mean, flow_std,
            )
            if ok:
                success += 1

        print(f"  {split}: {success}/{len(scene_ids)} scenes prepared")


if __name__ == "__main__":
    main()
