import argparse
import logging
from pathlib import Path

import numpy as np


DEFAULT_MULTICLASS_FEATURE_MODEL = "vrlens-finetunes-multiclass-v2-yolo11m-pose"
SUMMARY_FLOW_FILE = Path("flow/raft_f64_s0.5.npy")
DENSE_FLOW_FILE = Path("flow/raft_dense_32x32_s0.5.npy")

log = logging.getLogger(__name__)


def _resolve_processed_dir(data_dir: Path) -> Path:
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        return processed_dir
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embedding and flow normalization stats")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-model-name",
        type=str,
        default=DEFAULT_MULTICLASS_FEATURE_MODEL,
        help="Multiclass feature model stem used for embedding filenames",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz path for the computed statistics",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    processed_dir = _resolve_processed_dir(args.data_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    feature_model_name = Path(args.feature_model_name).stem
    if args.out is None:
        stats_dir = processed_dir.parent / "featurestats"
        if feature_model_name == DEFAULT_MULTICLASS_FEATURE_MODEL:
            args.out = stats_dir / "feature_stats.npz"
        else:
            args.out = stats_dir / f"feature_stats_{feature_model_name}.npz"
    emb_rel = Path("embeddings") / f"{feature_model_name}.npy"

    emb_sum: np.ndarray | None = None
    emb_sq_sum: np.ndarray | None = None
    flow_sum = np.zeros(64, dtype=np.float64)
    flow_sq_sum = np.zeros(64, dtype=np.float64)
    flow_dense_sum = np.zeros(2, dtype=np.float64)
    flow_dense_sq_sum = np.zeros(2, dtype=np.float64)
    n_emb_frames = 0
    n_flow_frames = 0
    n_dense_pixels = 0
    embeddings_found = 0
    summary_flow_found = 0
    dense_found = 0
    max_persons = 0
    embed_dim = 0

    scenes = [scene_dir for scene_dir in sorted(processed_dir.iterdir()) if scene_dir.is_dir()]
    for scene_dir in scenes:
        emb_path = scene_dir / emb_rel
        if not emb_path.exists():
            continue

        embeddings = np.load(str(emb_path), mmap_mode="r")
        if embeddings.ndim != 3:
            log.warning("Skipping %s due to unexpected embedding shape %s", scene_dir.name, embeddings.shape)
            continue

        emb_flat = embeddings.reshape(len(embeddings), -1).astype(np.float64)
        if emb_sum is None:
            emb_sum = np.zeros(emb_flat.shape[1], dtype=np.float64)
            emb_sq_sum = np.zeros(emb_flat.shape[1], dtype=np.float64)
            max_persons = int(embeddings.shape[1])
            embed_dim = int(embeddings.shape[2])
        elif emb_flat.shape[1] != emb_sum.shape[0]:
            log.warning(
                "Skipping %s due to embedding width mismatch %s vs expected %d",
                scene_dir.name,
                embeddings.shape,
                emb_sum.shape[0],
            )
            continue

        emb_sum += emb_flat.sum(axis=0)
        emb_sq_sum += np.square(emb_flat).sum(axis=0)
        n_emb_frames += len(emb_flat)
        embeddings_found += 1

        summary_flow_path = scene_dir / SUMMARY_FLOW_FILE
        if summary_flow_path.exists():
            summary_flow = np.load(str(summary_flow_path)).astype(np.float64)
            if summary_flow.ndim == 2 and summary_flow.shape[1] == 64:
                flow_sum += summary_flow.sum(axis=0)
                flow_sq_sum += np.square(summary_flow).sum(axis=0)
                n_flow_frames += len(summary_flow)
                summary_flow_found += 1
            else:
                log.warning(
                    "Skipping summary flow for %s due to unexpected shape %s",
                    scene_dir.name,
                    summary_flow.shape,
                )

        dense_flow_path = scene_dir / DENSE_FLOW_FILE
        if dense_flow_path.exists():
            dense_flow = np.load(str(dense_flow_path)).astype(np.float64)
            if dense_flow.ndim == 4 and dense_flow.shape[1] == 2:
                flow_dense_sum += dense_flow.sum(axis=(0, 2, 3))
                flow_dense_sq_sum += np.square(dense_flow).sum(axis=(0, 2, 3))
                n_dense_pixels += dense_flow.shape[0] * dense_flow.shape[2] * dense_flow.shape[3]
                dense_found += 1
            else:
                log.warning(
                    "Skipping dense flow for %s due to unexpected shape %s",
                    scene_dir.name,
                    dense_flow.shape,
                )

    if emb_sum is None or emb_sq_sum is None or n_emb_frames == 0:
        raise RuntimeError(f"No embeddings found for feature model {feature_model_name} under {processed_dir}")

    emb_mean = emb_sum / n_emb_frames
    emb_std = np.sqrt(np.maximum(emb_sq_sum / n_emb_frames - np.square(emb_mean), 1e-12))
    emb_std = np.maximum(emb_std, 1e-6)

    stats: dict[str, np.ndarray | int] = {
        "emb_mean": emb_mean.astype(np.float32),
        "emb_std": emb_std.astype(np.float32),
        "n_frames": int(n_emb_frames),
        "max_persons": int(max_persons),
        "embed_dim": int(embed_dim),
    }

    if n_flow_frames > 0:
        flow_mean = flow_sum / n_flow_frames
        flow_std = np.sqrt(np.maximum(flow_sq_sum / n_flow_frames - np.square(flow_mean), 1e-12))
        flow_std = np.maximum(flow_std, 1e-6)
        stats["flow_mean"] = flow_mean.astype(np.float32)
        stats["flow_std"] = flow_std.astype(np.float32)
        log.info("Summary flow found in %d/%d scenes", summary_flow_found, len(scenes))
        log.info("Flow mean range: [%.4f, %.4f]", flow_mean.min(), flow_mean.max())
        log.info("Flow std range: [%.4f, %.4f]", flow_std.min(), flow_std.max())
    else:
        log.warning("No summary flow files found; flow_mean/flow_std will be omitted")

    if n_dense_pixels > 0:
        flow_dense_mean = flow_dense_sum / n_dense_pixels
        flow_dense_std = np.sqrt(
            np.maximum(flow_dense_sq_sum / n_dense_pixels - np.square(flow_dense_mean), 1e-12)
        )
        flow_dense_std = np.maximum(flow_dense_std, 1e-6)
        stats["flow_dense_mean"] = flow_dense_mean.astype(np.float32).reshape(2, 1, 1)
        stats["flow_dense_std"] = flow_dense_std.astype(np.float32).reshape(2, 1, 1)
        log.info("Dense flow found in %d/%d scenes (%d pixels)", dense_found, len(scenes), n_dense_pixels)
        log.info("Dense flow mean: [%.6f, %.6f]", flow_dense_mean[0], flow_dense_mean[1])
        log.info("Dense flow std: [%.6f, %.6f]", flow_dense_std[0], flow_dense_std[1])
    else:
        log.warning("No dense flow files found; flow_dense_mean/flow_dense_std will be omitted")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **stats)

    log.info("Computed stats over %d embedding frames from %d/%d scenes", n_emb_frames, embeddings_found, len(scenes))
    log.info("Embedding width: %d persons x %d dims", max_persons, embed_dim)
    log.info("Emb mean range: [%.4f, %.4f]", emb_mean.min(), emb_mean.max())
    log.info("Emb std range: [%.4f, %.4f]", emb_std.min(), emb_std.max())
    log.info("Saved %s", args.out)


if __name__ == "__main__":
    main()