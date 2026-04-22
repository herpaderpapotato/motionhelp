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
import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.dispositiontcn import DispositionTCN, extract_disposition_config
from src.data.spatial import (
    legacy_conf_path,
    read_spatial_features_h5,
    read_spatial_metadata,
    spatial_feature_path,
)
from src.training.funscript_metrics import compute_regression_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "vrlens-finetunes-multiclass-v2-yolo26m-pose"
AUX_STATE_PREFIXES = (
    "aux_spatial_encoders",
    "aux_person_attn",
    "aux_proj",
    "aux_tcn_blocks",
    "aux_output_heads",
)


class SpatialDataset(Dataset):
    """Loads spatial RoI features + labels for DispositionTCN training.

    Expects:
        data/processed/{scene_id}/spatial/{model_name}.h5        [T, N, C, H, W] int8/float16
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
        device_dequantize: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.model_name = model_name
        self.augment = augment
        self.device_dequantize_requested = device_dequantize
        self.device_dequantize = False
        self._spatial_paths: dict[str, Path] = {}
        self._legacy_conf_paths: dict[str, Path | None] = {}
        self._label_paths: dict[str, Path] = {}
        self._spatial_files: dict[str, object] = {}
        self._legacy_conf_files: dict[str, object] = {}
        self._label_arrays: dict[str, np.memmap] = {}
        self._channel_scales: dict[str, np.ndarray] = {}
        self.spatial_metadata: dict[str, object] | None = None

        split_file = self.data_dir / "splits" / f"disposition_{split}.json"
        with open(split_file) as f:
            video_ids = json.load(f)

        self.sequences: list[tuple[str, int]] = []
        processed = self.data_dir / "processed"
        skipped = 0
        all_valid_caches_are_quantized = True

        for vid_id in video_ids:
            vid_dir = processed / vid_id
            spatial_path = spatial_feature_path(vid_dir, self.model_name)
            conf_path = legacy_conf_path(vid_dir, self.model_name)
            label_path = vid_dir / "labels.npy"

            if not spatial_path.exists() or not label_path.exists():
                skipped += 1
                continue

            try:
                with h5py.File(str(spatial_path), "r") as spatial_file:
                    metadata = read_spatial_metadata(spatial_path, file_handle=spatial_file)
                    if self.spatial_metadata is None:
                        self.spatial_metadata = metadata
                    has_conf = "conf" in spatial_file
                    storage_dtype = str(metadata.get("storage_dtype", spatial_file["spatial"].dtype))
                    if storage_dtype == "int8_qchannel" and "channel_scale" in spatial_file:
                        self._channel_scales[vid_id] = np.asarray(
                            spatial_file["channel_scale"][:], dtype=np.float32,
                        ).copy()
                    else:
                        # ignore
                        skipped += 1
                        continue

                        #all_valid_caches_are_quantized = False
            except Exception:
                skipped += 1
                continue

            if not has_conf and not conf_path.exists():
                skipped += 1
                continue

            self._spatial_paths[vid_id] = spatial_path
            self._legacy_conf_paths[vid_id] = conf_path if conf_path.exists() else None
            self._label_paths[vid_id] = label_path

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
        self.device_dequantize = (
            self.device_dequantize_requested
            and len(self._spatial_paths) > 0
            and all_valid_caches_are_quantized
        )
        if self.device_dequantize_requested and not self.device_dequantize:
            log.info(
                "SpatialDataset [%s]: disabling device-side dequantization because one or more caches are not int8_qchannel",
                split,
            )
        elif self.device_dequantize:
            log.info("SpatialDataset [%s]: device-side int8 dequantization enabled", split)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_spatial_files"] = {}
        state["_legacy_conf_files"] = {}
        state["_label_arrays"] = {}
        return state

    def close(self) -> None:
        for handle in self._spatial_files.values():
            handle.close()
        for handle in self._legacy_conf_files.values():
            handle.close()
        self._spatial_files.clear()
        self._legacy_conf_files.clear()
        self._label_arrays.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def sample_metadata(self) -> dict[str, object]:
        return dict(self.spatial_metadata or {})

    def _get_spatial_file(self, vid_id: str):
        spatial_path = self._spatial_paths[vid_id]
        cache_key = str(spatial_path)
        handle = self._spatial_files.get(cache_key)
        if handle is None:
            handle = h5py.File(str(spatial_path), "r")
            self._spatial_files[cache_key] = handle
        return handle

    def _get_legacy_conf_file(self, vid_id: str):
        conf_path = self._legacy_conf_paths.get(vid_id)
        if conf_path is None:
            return None
        cache_key = str(conf_path)
        handle = self._legacy_conf_files.get(cache_key)
        if handle is None:
            handle = h5py.File(str(conf_path), "r")
            self._legacy_conf_files[cache_key] = handle
        return handle

    def _get_label_array(self, vid_id: str) -> np.memmap:
        label_path = self._label_paths[vid_id]
        cache_key = str(label_path)
        handle = self._label_arrays.get(cache_key)
        if handle is None:
            handle = np.load(str(label_path), mmap_mode="r")
            self._label_arrays[cache_key] = handle
        return handle

    def _read_conf_slice(self, vid_id: str, sl: slice) -> np.ndarray:
        spatial_handle = self._get_spatial_file(vid_id)
        if "conf" in spatial_handle:
            return np.asarray(spatial_handle["conf"][sl], dtype=np.float32)

        legacy_handle = self._get_legacy_conf_file(vid_id)
        if legacy_handle is None:
            raise FileNotFoundError(
                f"No confidence dataset found for {vid_id} in {self._spatial_paths[vid_id]}"
            )
        return np.asarray(legacy_handle["conf"][sl], dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        end = start + self.seq_len
        sl = slice(start, end)

        labels = np.asarray(self._get_label_array(vid_id)[sl], dtype=np.float32).copy()

        if self.device_dequantize:
            spatial = np.asarray(self._get_spatial_file(vid_id)["spatial"][sl], dtype=np.int8)
            conf = self._read_conf_slice(vid_id, sl)
            batch = {
                "spatial": torch.from_numpy(spatial),
                "channel_scale": torch.from_numpy(self._channel_scales[vid_id]),
                "conf": torch.from_numpy(conf),
                "labels": torch.from_numpy(labels),
            }
        else:
            spatial, conf, _ = read_spatial_features_h5(
                self._spatial_paths[vid_id],
                start=start,
                end=end,
                file_handle=self._get_spatial_file(vid_id),
                legacy_conf_path=self._legacy_conf_paths.get(vid_id),
                legacy_conf_file_handle=self._get_legacy_conf_file(vid_id),
            )
            batch = {
                "spatial": torch.from_numpy(spatial),
                "conf": torch.from_numpy(conf),
                "labels": torch.from_numpy(labels),
            }

        spatial = batch["spatial"]
        conf = batch["conf"]
        labels = batch["labels"]

        ## I need to read the metrics file and look into the prepare_videos for actions-to-frame labels. It's essential to check label statistics across the data set and compute label distribution for the train/validation split using Python. Hmm, I should inspect the augmentation inversion's interaction too. Notably, the train_disposition randomly inverts labels without input changes, which seems suspicious! That could cause the output range to compress significantly. I need to check if this happens in train_tcn as well.
        # HOLY CRAP YOU NAILED IT!

        if self.augment:
            # Time reversal
            if torch.rand(1).item() < 0.3:
                spatial = spatial.flip(0)
                conf = conf.flip(0)
                labels = labels.flip(0)
            # # Position inversion  # KILLING THIS BECAUSE THIS IS REALLY BAD!
            # if torch.rand(1).item() < 0.3:
            #     labels = 1.0 - labels

        batch["spatial"] = spatial
        batch["conf"] = conf
        batch["labels"] = labels
        return batch

    def sample_random(self, n: int, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        idx = np.random.choice(len(self.sequences), min(n, len(self.sequences)), replace=False)
        batch = [self[i] for i in idx]
        out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        if device is not None:
            out = _prepare_batch(out, device, device_dequantize=self.device_dequantize)
        return out
    

def _resolve_sequence_limit(total_sequences: int, requested_limit: int | None) -> int:
    if requested_limit is None:
        return total_sequences
    return min(total_sequences, requested_limit)



class EpochSubsetSampler(Sampler[int]):
    # Re-samples a capped subset each epoch when shuffle=True.
    def __init__(
        self,
        dataset_size: int,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset_size = dataset_size
        self.max_samples = _resolve_sequence_limit(dataset_size, max_samples)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.max_samples

    def __iter__(self):
        if not self.shuffle:
            return iter(range(self.max_samples))

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator)[:self.max_samples]
        return iter(indices.tolist())


def _resolve_prefetch_factor(num_workers: int, requested_prefetch_factor: int | None) -> int | None:
    if num_workers <= 0:
        return None
    if requested_prefetch_factor is None:
        return 2
    return requested_prefetch_factor


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    sampler: Sampler[int] = None,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": sampler is None,  # shuffle if no sampler is provided
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def _prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    device_dequantize: bool,
) -> dict[str, torch.Tensor]:
    spatial_dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device_dequantize:
        spatial = batch["spatial"].to(device=device, dtype=spatial_dtype, non_blocking=True)
        channel_scale = batch["channel_scale"].to(
            device=device, dtype=spatial_dtype, non_blocking=True,
        )
        spatial = spatial * channel_scale[:, None, None, :, None, None]
    else:
        spatial = batch["spatial"].to(device=device, dtype=spatial_dtype, non_blocking=True)

    return {
        "spatial": spatial,
        "conf": batch["conf"].to(device=device, dtype=torch.float32, non_blocking=True),
        "labels": batch["labels"].to(device=device, dtype=torch.float32, non_blocking=True),
    }


def _scale_names_from_metadata(metadata: dict[str, object] | None) -> tuple[str, ...]:
    if not metadata:
        return ()
    scale_specs = metadata.get("scale_specs")
    if isinstance(scale_specs, list) and scale_specs:
        return tuple(
            str(spec.get("name", f"scale_{idx}"))
            for idx, spec in enumerate(scale_specs)
        )
    channel_slices = metadata.get("channel_slices")
    if isinstance(channel_slices, dict) and channel_slices:
        return tuple(str(name) for name in channel_slices.keys())
    return ()


def _split_predictions(pred: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if pred.ndim == 2:
        return pred, []
    if pred.ndim == 3:
        return pred[:, 0], [pred[:, idx] for idx in range(1, pred.shape[1])]
    raise ValueError(f"Unsupported prediction shape {tuple(pred.shape)}")


def _compute_loss(
    pred: torch.Tensor,
    lbl: torch.Tensor,
    pos_weight: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    main_pred, aux_preds = _split_predictions(pred)
    metrics = compute_regression_metrics(main_pred, lbl, spectral_kernel=15)

    loss = (
        metrics["pos_mse"] * pos_weight
        + args.velocity_weight * metrics["vel_mse"]
        + args.temporal_weight * metrics["acc_mse"]
    )

    if aux_preds:
        aux_branch_losses = []
        for idx, aux_pred in enumerate(aux_preds):
            branch_loss = F.mse_loss(aux_pred, lbl)
            aux_branch_losses.append(branch_loss)
            metrics[f"aux_branch_{idx}_mse"] = branch_loss.detach()
        aux_loss = torch.stack(aux_branch_losses).mean()
        metrics["aux_loss"] = aux_loss.detach()
        if args.aux_weight > 0:
            loss = loss + args.aux_weight * aux_loss

    return loss, main_pred, aux_preds, metrics


def _append_metric_history(
    history: dict[str, list[float]],
    metrics: dict[str, torch.Tensor],
) -> None:
    for key, value in metrics.items():
        history.setdefault(key, []).append(float(value.detach().item()))


def _write_prediction_overlay(
    writer: SummaryWriter,
    epoch: int,
    labels_np: np.ndarray,
    pred_np: np.ndarray,
    aux_pred_np: list[np.ndarray],
    aux_names: tuple[str, ...],
) -> None:
    n_plots = min(4, len(pred_np))
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots))
    if n_plots == 1:
        axes = [axes]

    aux_colors = ["forestgreen", "purple", "crimson", "slategray"]
    for i, ax in enumerate(axes):
        ax.plot(labels_np[i], label="target", alpha=0.85, lw=1.5, color="steelblue")
        ax.plot(pred_np[i], label="pred", alpha=0.9, lw=1.5, color="darkorange")
        for idx, aux_values in enumerate(aux_pred_np):
            aux_label = aux_names[idx] if idx < len(aux_names) else f"aux_{idx + 1}"
            ax.plot(
                aux_values[i],
                label=aux_label,
                alpha=0.45,
                lw=1.0,
                color=aux_colors[idx % len(aux_colors)],
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"#{i}", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)
            ax.set_title(f"Epoch {epoch} predictions")

    fig.tight_layout()
    writer.add_figure("Predictions/overlay", fig, epoch)
    plt.close(fig)


def _load_model_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    load_result = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in load_result.unexpected_keys
        if not key.startswith(AUX_STATE_PREFIXES)
    ]
    missing = [
        key for key in load_result.missing_keys
        if not key.startswith(AUX_STATE_PREFIXES)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state mismatch. "
            f"Missing keys: {missing or '[]'} | Unexpected keys: {unexpected or '[]'}"
        )
    aux_missing = [key for key in load_result.missing_keys if key.startswith(AUX_STATE_PREFIXES)]
    aux_unexpected = [key for key in load_result.unexpected_keys if key.startswith(AUX_STATE_PREFIXES)]
    if aux_missing:
        log.info("Initialising new aux weights not present in checkpoint: %s", ", ".join(aux_missing))
    if aux_unexpected:
        log.info("Ignoring aux weights from checkpoint: %s", ", ".join(aux_unexpected))




def train() -> None:
    parser = argparse.ArgumentParser(description="Train DispositionTCN")
    parser.set_defaults(shuffle=True, device_dequantize=None, use_aux_layers=False)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument(
        "--max-train-sequences-per-epoch",
        type=int,
        default=20000,
        help="Cap the number of train sequences used in each epoch",
    )
    parser.add_argument(
        "--max-val-sequences-per-epoch",
        type=int,
        default=None,
        help="Optional cap for validation sequences per epoch (default: 10%% of capped train set)",
    )
    parser.add_argument("--shuffle", action="store_true", dest="shuffle")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--n-persons", type=int, default=1)
    parser.add_argument("--use-ddl", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.45)
    parser.add_argument("--temporal-weight", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker when num_workers > 0",
    )
    parser.add_argument("--device-dequantize", action="store_true", dest="device_dequantize")
    parser.add_argument("--no-device-dequantize", action="store_false", dest="device_dequantize")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=None,
        help="Weight for aux branch MSE; defaults to 0.1 when --use-aux-layers is enabled",
    )
    parser.add_argument("--use-aux-layers", action="store_true", dest="use_aux_layers")
    parser.add_argument("--disable-aux-layers", action="store_false", dest="use_aux_layers")
    args = parser.parse_args()

    if args.max_train_sequences_per_epoch is not None and args.max_train_sequences_per_epoch <= 0:
        parser.error("--max-train-sequences-per-epoch must be a positive integer")
    if args.max_val_sequences_per_epoch is not None and args.max_val_sequences_per_epoch <= 0:
        parser.error("--max-val-sequences-per-epoch must be a positive integer")
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be a positive integer")
    if args.aux_weight is not None and args.aux_weight < 0:
        parser.error("--aux-weight must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    pos_weight = 1.0 - args.velocity_weight - args.temporal_weight
    if pos_weight < 0:
        pos_weight = 0.0
    if args.velocity_weight + args.temporal_weight > 1.0:
        total_weight = args.velocity_weight + args.temporal_weight  # e.g. 1.4
        args.velocity_weight = args.velocity_weight / total_weight # bring the total back to 1.0 while keeping the relative balance (e.g. 0.5 / 1.4)
        args.temporal_weight = args.temporal_weight / total_weight # (e.g. 0.9 / 1.4)
        pos_weight = 0.0


    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.device_dequantize is None:
        args.device_dequantize = device.type == "cuda"
    if args.aux_weight is None:
        args.aux_weight = 0.1 if args.use_aux_layers else 0.0
    if not args.use_aux_layers:
        args.aux_weight = 0.0

    prefetch_factor = _resolve_prefetch_factor(args.num_workers, args.prefetch_factor)

    # Datasets
    train_ds = SpatialDataset(
        args.data_dir, "train", args.seq_len, args.stride,
        model_name=args.model_name, augment=True,
        device_dequantize=args.device_dequantize,
    )
    val_ds = SpatialDataset(
        args.data_dir, "val", args.seq_len, args.stride,
        model_name=args.model_name, augment=False,
        device_dequantize=args.device_dequantize,
    )

    train_sequences_per_epoch = _resolve_sequence_limit(
        len(train_ds),
        args.max_train_sequences_per_epoch,
    )
    train_subset_is_limited = train_sequences_per_epoch < len(train_ds)
    val_sequences_per_epoch = args.max_val_sequences_per_epoch
    if val_sequences_per_epoch is None and train_subset_is_limited:
        val_sequences_per_epoch = min(len(val_ds), max(1, math.ceil(train_sequences_per_epoch * 0.1)))

    train_sampler = EpochSubsetSampler(
        len(train_ds),
        max_samples=train_sequences_per_epoch,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    val_sampler = EpochSubsetSampler(
        len(val_ds),
        max_samples=val_sequences_per_epoch,
        shuffle=False,
        seed=args.seed,
    )
    epoch_start = 0

    train_loader = _build_loader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor,
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor,
    )

    log.info(
        "Train: %d/%d sequences per epoch (%d batches, shuffle=%s, device_dequantize=%s)",
        len(train_sampler), len(train_ds), len(train_loader), args.shuffle, train_ds.device_dequantize,
    )
    if train_subset_is_limited or args.max_val_sequences_per_epoch is not None:
        log.info(
            "Val:   %d/%d sequences per epoch (%d batches)",
            len(val_sampler), len(val_ds), len(val_loader),
        )
    else:
        log.info("Val:   %d sequences (%d batches)", len(val_ds), len(val_loader))
    if prefetch_factor is not None:
        log.info("DataLoader prefetch_factor=%d with %d workers", prefetch_factor, args.num_workers)

    spatial_metadata = train_ds.sample_metadata()
    scale_channel_slices = (
        spatial_metadata.get("channel_slices")
        if isinstance(spatial_metadata.get("channel_slices"), dict)
        else None
    ) if spatial_metadata else None
    aux_scale_names = _scale_names_from_metadata(spatial_metadata)
    if args.use_aux_layers and not scale_channel_slices:
        raise ValueError(
            "--use-aux-layers requires channel_slices metadata in the spatial cache. "
            "Re-extract with the current scripts/extract_spatial.py first."
        )

    
    # train_loader = _build_loader(
    #     train_ds,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=device.type == "cuda",
    #     prefetch_factor=prefetch_factor,
    # )
    # val_loader = _build_loader(
    #     val_ds,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=device.type == "cuda",
    #     prefetch_factor=prefetch_factor,
    # )

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
        use_aux_layers=args.use_aux_layers,
        scale_channel_slices=scale_channel_slices,
        scale_names=aux_scale_names,
    ).to(device)

    params = model.count_parameters()
    log.info("Model: %s trainable / %s total parameters",
             f"{params['trainable']:,}", f"{params['total']:,}")
    if args.use_aux_layers:
        log.info(
            "Aux branches enabled for %s (aux_weight=%.3f)",
            ", ".join(model.aux_scale_names),
            args.aux_weight,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=max(1, args.epochs * len(train_loader) // 4),
    #     T_mult=1,
    #     eta_min=args.lr * 0.01,
    # )
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        _load_model_state(model, checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        log.info("Resumed training from checkpoint %s at epoch %d", args.resume, start_epoch)

        # initial validation
        # Validate
        model.eval()
        val_losses = []
        val_metrics = {"pos_mse": [], "vel_mse": [], "acc_mse": [], "vel_mae": []}

        with torch.no_grad():
            for batch in val_loader:
                prepared = _prepare_batch(batch, device, device_dequantize=val_ds.device_dequantize)
                spatial = prepared["spatial"]
                conf = prepared["conf"]
                lbl = prepared["labels"]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    raw_pred = model(spatial, conf)
                    loss, pred, _, metrics = _compute_loss(raw_pred, lbl, pos_weight, args)

                val_losses.append(loss.item())
                for key in val_metrics:
                    val_metrics[key].append(metrics[key].item())

        avg_val = np.mean(val_losses)
        avg_val_metrics = {k: float(np.mean(v)) for k, v in val_metrics.items()}
        epoch_time = time.time() - epoch_start
        log.info(
            "Epoch %d (resumed): train_loss=%.4f val_loss=%.4f pos_mse=%.4f vel_mse=%.4f acc_mse=%.4f vel_mae=%.4f time=%.1fs",
            start_epoch - 1, float("nan"), avg_val,
            avg_val_metrics["pos_mse"], avg_val_metrics["vel_mse"],
            avg_val_metrics["acc_mse"], avg_val_metrics["vel_mae"],
            epoch_time,
        )
        best_val_loss = avg_val
        early_stop_counter = 0


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
        "use_aux_layers": args.use_aux_layers,
        "scale_channel_slices": scale_channel_slices,
        "scale_names": list(aux_scale_names) if aux_scale_names else None,
    }
    data_config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "model_name": args.model_name,
        "roi_size": args.roi_size,
        "max_train_sequences_per_epoch": args.max_train_sequences_per_epoch,
        "max_val_sequences_per_epoch": args.max_val_sequences_per_epoch,
        "prefetch_factor": prefetch_factor,
        "device_dequantize": train_ds.device_dequantize,
        "aux_weight": args.aux_weight,
    }
    if spatial_metadata:
        data_config.update({
            "spatial_channel_slices": spatial_metadata.get("channel_slices"),
            "spatial_layer_indices": spatial_metadata.get("source_layers"),
            "spatial_strides": spatial_metadata.get("source_strides"),
            "spatial_format_version": spatial_metadata.get("format_version"),
            "spatial_storage_dtype": spatial_metadata.get("storage_dtype"),
            "spatial_scale_specs": spatial_metadata.get("scale_specs"),
        })

    # Training loop
    
    early_stop_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        sample_taken = False
        epoch_start = time.time()
        #train_sampler.set_epoch(epoch - 1)

        # Train
        model.train()
        train_losses = []
        train_metrics: dict[str, list[float]] = {"pos_mse": [], "vel_mse": [], "acc_mse": []}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
            prepared = _prepare_batch(batch, device, device_dequantize=train_ds.device_dequantize)
            spatial = prepared["spatial"]
            conf = prepared["conf"]
            lbl = prepared["labels"]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw_pred = model(spatial, conf)
                loss, pred, aux_preds, metrics = _compute_loss(raw_pred, lbl, pos_weight, args)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            _append_metric_history(train_metrics, metrics)
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/pos_mse", metrics["pos_mse"].item(), global_step)
                writer.add_scalar("train/vel_mse", metrics["vel_mse"].item(), global_step)
                writer.add_scalar("train/acc_mse", metrics["acc_mse"].item(), global_step)
                writer.add_scalar("train/pred_mean", pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                if "aux_loss" in metrics:
                    writer.add_scalar("train/aux_loss", metrics["aux_loss"].item(), global_step)
                    for idx, aux_pred in enumerate(aux_preds):
                        aux_name = aux_scale_names[idx] if idx < len(aux_scale_names) else f"aux_{idx + 1}"
                        writer.add_scalar(
                            f"train/aux_{aux_name}_mse",
                            metrics[f"aux_branch_{idx}_mse"].item(),
                            global_step,
                        )
                        writer.add_scalar(
                            f"train/aux_{aux_name}_mean",
                            aux_pred.mean().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            f"train/aux_{aux_name}_std",
                            aux_pred.std().item(),
                            global_step,
                        )

        avg_train = np.mean(train_losses)
        avg_train_metrics = {k: float(np.mean(v)) for k, v in train_metrics.items() if v}

        # Validate
        model.eval()
        val_losses = []
        val_metrics: dict[str, list[float]] = {"pos_mse": [], "vel_mse": [], "acc_mse": [], "vel_mae": []}
        
        with torch.no_grad():
            for batch in val_loader:
                prepared = _prepare_batch(batch, device, device_dequantize=val_ds.device_dequantize)
                spatial = prepared["spatial"]
                conf = prepared["conf"]
                lbl = prepared["labels"]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    raw_pred = model(spatial, conf)
                    loss, pred, _, metrics = _compute_loss(raw_pred, lbl, pos_weight, args)

                val_losses.append(loss.item())
                _append_metric_history(val_metrics, metrics)

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
        writer.add_scalar("val/acc_mse", avg_val_metrics["acc_mse"], epoch)
        writer.add_scalar("train/loss_epoch", avg_train, epoch)
        writer.add_scalar("train/pos_mse_epoch", avg_train_metrics["pos_mse"], epoch)
        writer.add_scalar("train/vel_mse_epoch", avg_train_metrics["vel_mse"], epoch)
        writer.add_scalar("train/acc_mse_epoch", avg_train_metrics["acc_mse"], epoch)
        if "aux_loss" in avg_val_metrics:
            writer.add_scalar("val/aux_loss", avg_val_metrics["aux_loss"], epoch)
        if "aux_loss" in avg_train_metrics:
            writer.add_scalar("train/aux_loss_epoch", avg_train_metrics["aux_loss"], epoch)
            for idx, aux_name in enumerate(aux_scale_names):
                train_key = f"aux_branch_{idx}_mse"
                val_key = f"aux_branch_{idx}_mse"
                if train_key in avg_train_metrics:
                    writer.add_scalar(f"train/aux_{aux_name}_mse_epoch", avg_train_metrics[train_key], epoch)
                if val_key in avg_val_metrics:
                    writer.add_scalar(f"val/aux_{aux_name}_mse", avg_val_metrics[val_key], epoch)

        # Prediction plots every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            sample_taken = True
            model.eval()
            with torch.no_grad():
                if epoch == 1:
                    plot_samples = val_ds.sample_random(n=4, device=device)
                sp = plot_samples["spatial"]
                co = plot_samples["conf"]
                lb = plot_samples["labels"]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred_s = model(sp, co)

                main_pred_s, aux_pred_s = _split_predictions(pred_s)
                pred_s = main_pred_s.float().cpu().numpy()
                aux_pred_np = [aux.float().cpu().numpy() for aux in aux_pred_s]
                lb_np = lb.float().cpu().numpy()

            _write_prediction_overlay(writer, epoch, lb_np, pred_s, aux_pred_np, aux_scale_names)

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
            if not sample_taken:
                # take a sample for prediction plotting if we haven't already this epoch
                model.eval()
                with torch.no_grad():
                    if epoch == 1:
                        plot_samples = val_ds.sample_random(n=4, device=device)
                    sp = plot_samples["spatial"]
                    co = plot_samples["conf"]
                    lb = plot_samples["labels"]

                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        pred_s = model(sp, co)

                    main_pred_s, aux_pred_s = _split_predictions(pred_s)
                    pred_s = main_pred_s.float().cpu().numpy()
                    aux_pred_np = [aux.float().cpu().numpy() for aux in aux_pred_s]
                    lb_np = lb.float().cpu().numpy()
                _write_prediction_overlay(writer, epoch, lb_np, pred_s, aux_pred_np, aux_scale_names)
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
