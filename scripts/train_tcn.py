"""Training script for the TCN funscript prediction model.

Usage:
    python scripts/train_tcn.py --epochs 100
    python scripts/train_tcn.py --epochs 5 --batch-size 32  # quick test

Loads pre-extracted features (keypoints, embeddings, flow) from data/processed/
and trains a temporal convolutional network for per-frame position prediction.
"""

import argparse
from dataclasses import dataclass, replace
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
from src.models.tcn import FunscriptTCN, extract_model_config
from src.training.funscript_metrics import compute_regression_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

log = logging.getLogger(__name__)


REGRESSION_METRIC_KEYS = (
    "pos_mse",
    "event_mse",
    "active_mse",
    "vel_mse",
    "vel_mae",
    "acc_mse",
    "acc_mae",
    "spec_mse",
)

MODALITY_NAMES = ("pose", "embedding", "flow")


@dataclass(frozen=True)
class LossConfig:
    event_weight: float
    event_activity_gain: float
    event_activity_power: float
    active_quantile: float
    temporal_weight: float
    velocity_weight: float
    spectral_weight: float
    spectral_kernel: int


@dataclass(frozen=True)
class ModalityMask:
    drop_pose: bool = False
    drop_embedding: bool = False
    drop_flow: bool = False


@dataclass(frozen=True)
class ModalityDropoutConfig:
    pose_prob: float = 0.0
    embedding_prob: float = 0.0
    flow_prob: float = 0.0
    scale_with_augment: bool = False
    keep_at_least_one: bool = True

    def is_enabled(self) -> bool:
        return any(prob > 0.0 for prob in (self.pose_prob, self.embedding_prob, self.flow_prob))

    def scaled(self, augment_scale: float) -> "ModalityDropoutConfig":
        if not self.scale_with_augment:
            return self
        scale = 1.0 + max(0.0, augment_scale)
        return ModalityDropoutConfig(
            pose_prob=min(1.0, self.pose_prob * scale),
            embedding_prob=min(1.0, self.embedding_prob * scale),
            flow_prob=min(1.0, self.flow_prob * scale),
            scale_with_augment=self.scale_with_augment,
            keep_at_least_one=self.keep_at_least_one,
        )


@dataclass(frozen=True)
class DominanceControlConfig:
    response: str = "none"
    dominance_threshold: float = 0.55
    min_reliance_delta: float = 1e-4
    gradient_weight: float = 0.3
    lr_decay: float = 0.8
    min_lr_scale: float = 0.1
    dropout_increase: float = 0.05
    max_dropout_prob: float = 0.5

    def enabled(self) -> bool:
        return self.response != "none"


def _apply_modality_mask(
    keypoints: torch.Tensor,
    embeddings: torch.Tensor,
    flow: torch.Tensor,
    mask: ModalityMask,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask.drop_pose:
        keypoints = torch.zeros_like(keypoints)
    if mask.drop_embedding:
        embeddings = torch.zeros_like(embeddings)
    if mask.drop_flow:
        flow = torch.zeros_like(flow)
    return keypoints, embeddings, flow


def _sample_modality_mask(
    dropout_config: ModalityDropoutConfig,
    augment_scale: float,
) -> ModalityMask:
    if not dropout_config.is_enabled():
        return ModalityMask()

    scaled_config = dropout_config.scaled(augment_scale)
    drop_pose = torch.rand(()).item() < scaled_config.pose_prob
    drop_embedding = torch.rand(()).item() < scaled_config.embedding_prob
    drop_flow = torch.rand(()).item() < scaled_config.flow_prob

    if scaled_config.keep_at_least_one and drop_pose and drop_embedding and drop_flow:
        keep_idx = int(torch.randint(0, 3, (1,)).item())
        if keep_idx == 0:
            drop_pose = False
        elif keep_idx == 1:
            drop_embedding = False
        else:
            drop_flow = False

    return ModalityMask(
        drop_pose=drop_pose,
        drop_embedding=drop_embedding,
        drop_flow=drop_flow,
    )


def _init_metric_history() -> dict[str, list[float]]:
    return {key: [] for key in REGRESSION_METRIC_KEYS}


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
    DENSE_FLOW_FILE = "flow/raft_dense_32x32_s0.5.npy"

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
        modality_dropout: ModalityDropoutConfig | None = None,
        flow_mode: str = "summary",
        flow_dense_size: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.n_persons = n_persons
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim
        self.augment = augment
        self.multiclass = multiclass
        self.modality_dropout = modality_dropout or ModalityDropoutConfig()
        self.flow_mode = flow_mode
        self.flow_dense_size = flow_dense_size
        self.using_stats = False
        self.stats_path = None
        self.runs = 0

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
        self.augment_scale: float = 0.0  # 0=min regularization, 1=max regularization

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

            if self.flow_mode == "dense":
                dense_flow_p = vid_dir / self.DENSE_FLOW_FILE
                required = [kp_path, emb_path, emb_meta_path, dense_flow_p, label_path]
            else:
                required = [kp_path, emb_path, emb_meta_path, flow_path, label_path]
            if not all(p.exists() for p in required):
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
        stats_path = self.data_dir / "featurestats" / "feature_stats.npz"
        self.using_stats = False
        if not stats_path.exists():
            log.warning("No feature_stats.npz — features will NOT be normalized")
            self.emb_mean = self.emb_std = None
            self.flow_mean = self.flow_std = None
            return
        self.stats_path = stats_path
        self.using_stats = True

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
            
            self.using_stats = False
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
            
            self.using_stats = False
            log.warning("Flow stats not found or shape mismatch — not normalizing flow")

        # Dense flow stats
        self.flow_dense_mean = None
        self.flow_dense_std = None
        if self.flow_mode == "dense":
            flow_dense_mean = stats.get("flow_dense_mean")
            flow_dense_std = stats.get("flow_dense_std")
            if flow_dense_mean is not None:
                self.flow_dense_mean = flow_dense_mean  # [2, 1, 1]
                self.flow_dense_std = flow_dense_std
                log.info("Loaded dense flow normalization stats: shape=%s from %s",
                         flow_dense_mean.shape, stats_path)
            else:
                log.warning("Dense flow stats not found — dense flow will NOT be normalized")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_id, start = self.sequences[idx]
        vid_dir = self.data_dir / "processed" / vid_id
        end = start + self.seq_len

        labels = np.load(str(vid_dir / "labels.npy"), mmap_mode="r")[start:end].copy()
        keypoints = np.load(str(vid_dir / self.KP_FILE), mmap_mode="r")[start:end].copy()
        embeddings = np.load(str(vid_dir / self.EMB_FILE), mmap_mode="r")[start:end].copy()

        # Normalize embeddings
        if self.emb_mean is not None:
            embeddings = (embeddings - self.emb_mean) / (self.emb_std + 1e-8)

        if self.flow_mode == "dense":
            flow = np.load(str(vid_dir / self.DENSE_FLOW_FILE), mmap_mode="r")[start:end].copy().astype(np.float32)
            # Normalize dense flow: [T, 2, H, W] with stats [2, 1, 1]
            if self.flow_dense_mean is not None:
                flow = (flow - self.flow_dense_mean) / (self.flow_dense_std + 1e-8)
        else:
            flow = np.load(str(vid_dir / self.FLOW_FILE), mmap_mode="r")[start:end].copy()
            if self.flow_mean is not None:
                flow = (flow - self.flow_mean) / (self.flow_std + 1e-8)

        keypoints = torch.from_numpy(keypoints).float()    # [T, N, K, 3]
        embeddings = torch.from_numpy(embeddings).float()   # [T, N, E]
        flow = torch.from_numpy(flow).float()               # [T, F]
        labels = torch.from_numpy(labels).float()            # [T]

        if self.augment:
            keypoints, embeddings, flow = self._augment(keypoints, embeddings, flow)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Randomly blank whole modalities to prevent the network from overusing one path.
        mask = _sample_modality_mask(self.modality_dropout, self.augment_scale)
        return _apply_modality_mask(kp, emb, flow, mask)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _compute_augment_scale(val_loss: float, train_loss: float) -> float:
    """Map val/train loss ratio to regularization scale [0, 1].

    Linear interpolation in between.
    """
    start_threshold = 1.0
    end_threshold = 2.0
    if train_loss <= 0:
        return 0.0
    ratio = val_loss / train_loss
    if ratio < start_threshold:
        return 0.0
    if ratio >= end_threshold:
        return 1.0
    return (ratio - start_threshold) / (end_threshold - start_threshold)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _resolve_probability(override: float | None, base: float) -> float:
    return base if override is None else override


def _dropout_field_name(modality: str) -> str:
    if modality == "pose":
        return "pose_prob"
    if modality == "embedding":
        return "embedding_prob"
    if modality == "flow":
        return "flow_prob"
    raise ValueError(f"Unsupported modality: {modality}")


def _compute_batch_loss(
    model: FunscriptTCN,
    keypoints: torch.Tensor,
    embeddings: torch.Tensor,
    flow: torch.Tensor,
    labels: torch.Tensor,
    loss_config: LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pred = model(keypoints, embeddings, flow)
    metric_batch = compute_regression_metrics(
        pred,
        labels,
        spectral_kernel=loss_config.spectral_kernel,
        activity_gain=loss_config.event_activity_gain,
        activity_power=loss_config.event_activity_power,
        active_quantile=loss_config.active_quantile,
    )
    pos_loss = ((1.0 - loss_config.event_weight) * metric_batch["pos_mse"]
                + loss_config.event_weight * metric_batch["event_mse"])
    temporal_loss = metric_batch["acc_mse"]
    velocity_loss = metric_batch["vel_mse"]
    spectral_loss = metric_batch["spec_mse"]
    loss = (pos_loss
            + loss_config.temporal_weight * temporal_loss
            + loss_config.velocity_weight * velocity_loss
            + loss_config.spectral_weight * spectral_loss)
    return pred, loss, metric_batch, {
        "pos_loss": pos_loss,
        "temp_loss": temporal_loss,
        "vel_loss": velocity_loss,
        "spec_loss": spectral_loss,
    }


def _build_optimizer_param_groups(
    model: FunscriptTCN,
    base_lr: float,
    weight_decay: float,
    pose_lr_scale: float,
    embedding_lr_scale: float,
    flow_lr_scale: float,
    aux_lr_scale: float,
    shared_lr_scale: float,
) -> list[dict[str, object]]:
    param_groups: list[dict[str, object]] = []
    assigned_param_ids: set[int] = set()

    def add_group(name: str, modules: list[nn.Module | None], lr_scale: float) -> None:
        params: list[nn.Parameter] = []
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad or id(param) in assigned_param_ids:
                    continue
                params.append(param)
                assigned_param_ids.add(id(param))
        if params:
            param_groups.append({
                "name": name,
                "params": params,
                "lr": base_lr * lr_scale,
                "weight_decay": weight_decay,
            })

    pose_modules: list[nn.Module | None] = [model.pose_encoder, model.pose_attn]
    if model.multiclass:
        pose_modules.append(model.beholder_pose_encoder)
    add_group("pose", pose_modules, pose_lr_scale)

    embedding_modules: list[nn.Module | None] = [model.emb_encoder, model.emb_attn]
    if model.multiclass:
        embedding_modules.append(model.beholder_emb_encoder)
    add_group("embedding", embedding_modules, embedding_lr_scale)

    add_group("flow", [model.flow_encoder], flow_lr_scale)

    aux_modules: list[nn.Module | None] = []
    if getattr(model, "use_kinematics", False):
        aux_modules.extend([getattr(model, "kin_encoder", None), getattr(model, "kin_attn", None)])
    if getattr(model, "use_difference_pathway", False):
        aux_modules.append(getattr(model, "difference_encoder", None))
    add_group("aux", aux_modules, aux_lr_scale)

    add_group("shared", [model.fusion, model.tcn_blocks, model.output_head], shared_lr_scale)

    remaining_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in assigned_param_ids
    ]
    if remaining_params:
        log.warning("Assigning %d ungrouped parameters to optimizer fallback group", len(remaining_params))
        param_groups.append({
            "name": "remaining",
            "params": remaining_params,
            "lr": base_lr * shared_lr_scale,
            "weight_decay": weight_decay,
        })

    return param_groups


def _normalize_modality_shares(shares: dict[str, float]) -> dict[str, float]:
    raw = {name: max(float(shares.get(name, 0.0)), 0.0) for name in MODALITY_NAMES}
    total = sum(raw.values())
    if total <= 0.0:
        return {name: 0.0 for name in MODALITY_NAMES}
    return {name: value / total for name, value in raw.items()}


def _apply_live_lr_scales(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    base_lr: float,
    live_lr_scales: dict[str, float],
) -> None:
    for group_index, param_group in enumerate(optimizer.param_groups):
        group_name = str(param_group.get("name", "group"))
        if group_name not in live_lr_scales:
            continue

        new_lr = base_lr * live_lr_scales[group_name]
        param_group["lr"] = new_lr
        param_group["initial_lr"] = new_lr

        if hasattr(scheduler, "base_lrs") and group_index < len(scheduler.base_lrs):
            scheduler.base_lrs[group_index] = new_lr
        if hasattr(scheduler, "max_lrs") and group_index < len(scheduler.max_lrs):
            scheduler.max_lrs[group_index] = max(new_lr, 1e-12)


def _refresh_optimization_config(
    optimization_config: dict[str, object],
    live_lr_scales: dict[str, float],
    modality_dropout: ModalityDropoutConfig,
    last_action: dict[str, object] | None,
) -> None:
    optimization_config["current_lr_scales"] = dict(live_lr_scales)
    optimization_config["modality_dropout"] = {
        "pose_prob": modality_dropout.pose_prob,
        "embedding_prob": modality_dropout.embedding_prob,
        "flow_prob": modality_dropout.flow_prob,
        "scale_with_augment": modality_dropout.scale_with_augment,
    }
    optimization_config["last_dominance_action"] = last_action


def _collect_gradient_group_stats(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, float], dict[str, float]]:
    grad_sq_sums: dict[str, float] = {}
    total_sq = 0.0

    for group in optimizer.param_groups:
        group_name = str(group.get("name", "group"))
        group_sq = 0.0
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            group_sq += float(torch.sum(grad * grad).item())
        grad_sq_sums[group_name] = group_sq
        total_sq += group_sq

    grad_norms = {name: value ** 0.5 for name, value in grad_sq_sums.items()}
    grad_shares = {
        name: (value / total_sq if total_sq > 0.0 else 0.0)
        for name, value in grad_sq_sums.items()
    }
    return grad_norms, grad_shares


def _run_validation_epoch(
    model: FunscriptTCN,
    data_loader: DataLoader,
    device: torch.device,
    loss_config: LossConfig,
    modality_mask: ModalityMask | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float], float, float]:
    model.eval()
    losses: list[float] = []
    pred_means: list[float] = []
    pred_stds: list[float] = []
    metric_history = _init_metric_history()

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            kp = batch["keypoints"].to(device, non_blocking=True)
            emb = batch["embeddings"].to(device, non_blocking=True)
            fl = batch["flow"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            if modality_mask is not None:
                kp, emb, fl = _apply_modality_mask(kp, emb, fl, modality_mask)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred, loss, metric_batch, _ = _compute_batch_loss(
                    model,
                    kp,
                    emb,
                    fl,
                    lbl,
                    loss_config,
                )

            losses.append(float(loss.item()))
            for key in metric_history:
                metric_history[key].append(float(metric_batch[key].item()))
            pred_means.append(float(pred.mean().item()))
            pred_stds.append(float(pred.std().item()))

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    avg_metrics = {
        key: (float(np.mean(values)) if values else float("nan"))
        for key, values in metric_history.items()
    }
    avg_pred_mean = float(np.mean(pred_means)) if pred_means else float("nan")
    avg_pred_std = float(np.mean(pred_stds)) if pred_stds else float("nan")
    return avg_loss, avg_metrics, avg_pred_mean, avg_pred_std


def _estimate_modality_reliance(
    model: FunscriptTCN,
    data_loader: DataLoader,
    device: torch.device,
    loss_config: LossConfig,
    max_batches: int,
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
    baseline_loss, _, _, _ = _run_validation_epoch(
        model,
        data_loader,
        device,
        loss_config,
        max_batches=max_batches,
    )
    ablated_losses: dict[str, float] = {}
    reliance_delta: dict[str, float] = {}
    modality_masks = {
        "pose": ModalityMask(drop_pose=True),
        "embedding": ModalityMask(drop_embedding=True),
        "flow": ModalityMask(drop_flow=True),
    }
    for name, mask in modality_masks.items():
        ablated_loss, _, _, _ = _run_validation_epoch(
            model,
            data_loader,
            device,
            loss_config,
            modality_mask=mask,
            max_batches=max_batches,
        )
        ablated_losses[name] = ablated_loss
        reliance_delta[name] = ablated_loss - baseline_loss

    positive_delta = {name: max(delta, 0.0) for name, delta in reliance_delta.items()}
    total_positive = sum(positive_delta.values())
    reliance_share = {
        name: (value / total_positive if total_positive > 0.0 else 0.0)
        for name, value in positive_delta.items()
    }
    return baseline_loss, ablated_losses, reliance_delta, reliance_share


def _select_dominant_modality(
    normalized_gradient_share: dict[str, float],
    reliance_delta: dict[str, float],
    reliance_share: dict[str, float],
    control_config: DominanceControlConfig,
) -> tuple[str | None, dict[str, float], str]:
    combined_scores = {
        name: ((1.0 - control_config.gradient_weight) * reliance_share.get(name, 0.0)
               + control_config.gradient_weight * normalized_gradient_share.get(name, 0.0))
        for name in MODALITY_NAMES
    }
    dominant_modality = max(combined_scores, key=combined_scores.get)
    dominant_delta = reliance_delta.get(dominant_modality, 0.0)
    dominant_score = combined_scores[dominant_modality]

    if dominant_delta < control_config.min_reliance_delta:
        return None, combined_scores, (
            f"dominant delta {dominant_delta:.6f} is below min_reliance_delta={control_config.min_reliance_delta:.6f}"
        )
    if dominant_score < control_config.dominance_threshold:
        return None, combined_scores, (
            f"dominant score {dominant_score:.3f} is below dominance_threshold={control_config.dominance_threshold:.3f}"
        )
    return dominant_modality, combined_scores, "ok"


def _apply_dominance_response(
    dominant_modality: str,
    control_config: DominanceControlConfig,
    train_ds: MotionDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    base_lr: float,
    live_lr_scales: dict[str, float],
) -> dict[str, object]:
    action: dict[str, object] = {
        "mode": control_config.response,
        "dominant_modality": dominant_modality,
        "lr_scale_change": None,
        "dropout_change": None,
    }

    if control_config.response in {"lr", "both"} and dominant_modality in live_lr_scales:
        old_scale = live_lr_scales[dominant_modality]
        new_scale = max(control_config.min_lr_scale, old_scale * control_config.lr_decay)
        if new_scale < old_scale:
            live_lr_scales[dominant_modality] = new_scale
            _apply_live_lr_scales(optimizer, scheduler, base_lr, live_lr_scales)
            action["lr_scale_change"] = {
                "old": old_scale,
                "new": new_scale,
            }

    if control_config.response in {"dropout", "both"}:
        field_name = _dropout_field_name(dominant_modality)
        old_prob = float(getattr(train_ds.modality_dropout, field_name))
        new_prob = min(control_config.max_dropout_prob, old_prob + control_config.dropout_increase)
        if new_prob > old_prob:
            train_ds.modality_dropout = replace(train_ds.modality_dropout, **{field_name: new_prob})
            action["dropout_change"] = {
                "old": old_prob,
                "new": new_prob,
            }

    return action


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    import random
    parser = argparse.ArgumentParser(description="Train TCN funscript model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pose-lr-scale", type=float, default=1.0,
                        help="Learning-rate multiplier for pose/keypoint branches")
    parser.add_argument("--embedding-lr-scale", type=float, default=1.0,
                        help="Learning-rate multiplier for embedding branches")
    parser.add_argument("--flow-lr-scale", type=float, default=1.0,
                        help="Learning-rate multiplier for flow branch")
    parser.add_argument("--aux-lr-scale", type=float, default=1.0,
                        help="Learning-rate multiplier for auxiliary branches such as kinematics")
    parser.add_argument("--shared-lr-scale", type=float, default=1.0,
                        help="Learning-rate multiplier for fusion, TCN blocks, and output head")
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="CosineWarmupLR",
                        choices=["OneCycleLR", "CosineAnnealingLR", "CosineWarmupLR", "ReduceLROnPlateau"])
    parser.add_argument("--multiclass", action="store_true",
                        help="Use multiclass model (partner + beholder)")
    parser.add_argument("--n-partners", type=int, default=5)
    parser.add_argument("--n-beholders", type=int, default=1)
    parser.add_argument("--n-beholder-keypoints", type=int, default=7)
    parser.add_argument("--use-kinematics", action="store_true",
                        help="Add velocity/acceleration features from keypoints")
    parser.add_argument("--use-ddl", action="store_true",
                        help="Use Dual Dilated Layers (MS-TCN++ style)")
    parser.add_argument("--kin-dim", type=int, default=64,
                        help="Kinematic feature dimension")
    parser.add_argument("--use-gated-fusion", action="store_true",
                        help="Use context-aware gated multimodal fusion instead of concat+linear")
    parser.add_argument("--no-difference-pathway", action="store_true", default=False,
                        help="Disable the multiclass beholder-performer difference branch")
    parser.add_argument("--difference-dim", type=int, default=64,
                        help="Feature dimension for the beholder-performer difference branch")
    parser.add_argument("--velocity-weight", type=float, default=0.0,
                        help="Weight for velocity-matching loss (first derivative)")
    parser.add_argument("--spectral-weight", type=float, default=0.0,
                        help="Weight for high-frequency detail loss (multi-scale spectral)")
    parser.add_argument("--spectral-kernel", type=int, default=15,
                        help="Moving-average kernel size for spectral loss low-pass filter")
    parser.add_argument("--event-weight", type=float, default=0.25,
                        help="Blend factor between plain MSE and event-aware weighted MSE")
    parser.add_argument("--event-activity-gain", type=float, default=3.0,
                        help="Gain applied to derivative-based event weighting")
    parser.add_argument("--event-activity-power", type=float, default=1.0,
                        help="Power applied to derivative-based event weighting")
    parser.add_argument("--active-quantile", type=float, default=0.8,
                        help="Quantile threshold used when reporting active-frame metrics")
    parser.add_argument("--modality-dropout-prob", type=float, default=0.0,
                        help="Base independent dropout probability for each modality during training")
    parser.add_argument("--pose-dropout-prob", type=float, default=None,
                        help="Override dropout probability for pose/keypoint inputs")
    parser.add_argument("--embedding-dropout-prob", type=float, default=None,
                        help="Override dropout probability for embedding inputs")
    parser.add_argument("--flow-dropout-prob", type=float, default=None,
                        help="Override dropout probability for flow inputs")
    parser.add_argument("--scale-modality-dropout-with-augment", action="store_true", default=False,
                        help="Increase modality dropout when the validation/train loss gap grows")
    parser.add_argument("--track-modality-dominance", action="store_true", default=False,
                        help="Log per-branch gradient share and validation ablation deltas")
    parser.add_argument("--dominance-eval-every", type=int, default=1,
                        help="Run validation ablations every N epochs when tracking modality dominance")
    parser.add_argument("--dominance-max-batches", type=int, default=4,
                        help="Use this many validation batches for each dominance-ablation pass")
    parser.add_argument("--dominance-response", type=str, default="none",
                        choices=["none", "lr", "dropout", "both"],
                        help="How to react online when a modality dominates")
    parser.add_argument("--dominance-threshold", type=float, default=0.55,
                        help="Minimum combined dominance score required before applying an online response")
    parser.add_argument("--dominance-min-delta", type=float, default=1e-4,
                        help="Minimum positive validation ablation delta before a modality can be treated as dominant")
    parser.add_argument("--dominance-gradient-weight", type=float, default=0.3,
                        help="Weight given to normalized gradient share when scoring modality dominance")
    parser.add_argument("--dominance-lr-decay", type=float, default=0.8,
                        help="Multiply the dominant modality LR scale by this factor when dominance-response includes lr")
    parser.add_argument("--dominance-min-lr-scale", type=float, default=0.1,
                        help="Lower bound for live modality LR scale when dominance-response includes lr")
    parser.add_argument("--dominance-dropout-step", type=float, default=0.05,
                        help="Increase the dominant modality dropout probability by this amount when dominance-response includes dropout")
    parser.add_argument("--dominance-max-dropout", type=float, default=0.5,
                        help="Upper bound for live modality dropout probability when dominance-response includes dropout")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop training if val loss has not improved for this many epochs (0 = disabled)")
    parser.add_argument("--load-best-val-loss", action="store_true", default=False,
                        help="When resuming, load the best_val_loss from the checkpoint to continue early stopping correctly")
    parser.add_argument("--flow-mode", type=str, default="summary",
                        choices=["summary", "dense"],
                        help="Flow representation: 'summary' (flat 64-d) or 'dense' (2×32×32 spatial)")
    parser.add_argument("--flow-dense-size", type=int, default=32,
                        help="Spatial resolution for dense flow maps (default: 32)")
    args = parser.parse_args()
    args.stride = max(1, min(args.stride, args.seq_len//2))  # Ensure stride is at least 1 and at most seq_len
    if not 0.0 <= args.event_weight <= 1.0:
        parser.error("--event-weight must be between 0 and 1")
    if not 0.0 < args.active_quantile <= 1.0:
        parser.error("--active-quantile must be in (0, 1]")
    for arg_name, value in {
        "--modality-dropout-prob": args.modality_dropout_prob,
        "--pose-dropout-prob": args.pose_dropout_prob,
        "--embedding-dropout-prob": args.embedding_dropout_prob,
        "--flow-dropout-prob": args.flow_dropout_prob,
    }.items():
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"{arg_name} must be between 0 and 1")
    for arg_name, value in {
        "--pose-lr-scale": args.pose_lr_scale,
        "--embedding-lr-scale": args.embedding_lr_scale,
        "--flow-lr-scale": args.flow_lr_scale,
        "--aux-lr-scale": args.aux_lr_scale,
        "--shared-lr-scale": args.shared_lr_scale,
    }.items():
        if value <= 0.0:
            parser.error(f"{arg_name} must be greater than 0")
    if args.dominance_eval_every <= 0:
        parser.error("--dominance-eval-every must be greater than 0")
    if args.dominance_max_batches <= 0:
        parser.error("--dominance-max-batches must be greater than 0")
    if not 0.0 <= args.dominance_threshold <= 1.0:
        parser.error("--dominance-threshold must be between 0 and 1")
    if args.dominance_min_delta < 0.0:
        parser.error("--dominance-min-delta must be non-negative")
    if not 0.0 <= args.dominance_gradient_weight <= 1.0:
        parser.error("--dominance-gradient-weight must be between 0 and 1")
    if not 0.0 < args.dominance_lr_decay <= 1.0:
        parser.error("--dominance-lr-decay must be in (0, 1]")
    if args.dominance_min_lr_scale <= 0.0:
        parser.error("--dominance-min-lr-scale must be greater than 0")
    if args.dominance_max_dropout < 0.0 or args.dominance_max_dropout > 1.0:
        parser.error("--dominance-max-dropout must be between 0 and 1")
    if args.dominance_dropout_step < 0.0:
        parser.error("--dominance-dropout-step must be non-negative")
    if args.scheduler == "OneCycleLR" and args.dominance_response in {"lr", "both"}:
        parser.error("--dominance-response lr/both is not supported with OneCycleLR; use dropout/none or change scheduler")

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
        log.info("Setting CUDA seeds for reproducibility (seed=%d)", args.seed)
        torch.cuda.manual_seed_all(args.seed)

    loss_config = LossConfig(
        event_weight=args.event_weight,
        event_activity_gain=args.event_activity_gain,
        event_activity_power=args.event_activity_power,
        active_quantile=args.active_quantile,
        temporal_weight=args.temporal_weight,
        velocity_weight=args.velocity_weight,
        spectral_weight=args.spectral_weight,
        spectral_kernel=args.spectral_kernel,
    )
    modality_dropout_config = ModalityDropoutConfig(
        pose_prob=_resolve_probability(args.pose_dropout_prob, args.modality_dropout_prob),
        embedding_prob=_resolve_probability(args.embedding_dropout_prob, args.modality_dropout_prob),
        flow_prob=_resolve_probability(args.flow_dropout_prob, args.modality_dropout_prob),
        scale_with_augment=args.scale_modality_dropout_with_augment,
    )
    dominance_control = DominanceControlConfig(
        response=args.dominance_response,
        dominance_threshold=args.dominance_threshold,
        min_reliance_delta=args.dominance_min_delta,
        gradient_weight=args.dominance_gradient_weight,
        lr_decay=args.dominance_lr_decay,
        min_lr_scale=args.dominance_min_lr_scale,
        dropout_increase=args.dominance_dropout_step,
        max_dropout_prob=args.dominance_max_dropout,
    )
    track_modality_dominance = args.track_modality_dominance or dominance_control.enabled()
    if modality_dropout_config.is_enabled():
        log.info(
            "Modality dropout enabled: pose=%.3f embedding=%.3f flow=%.3f%s",
            modality_dropout_config.pose_prob,
            modality_dropout_config.embedding_prob,
            modality_dropout_config.flow_prob,
            " (scaled by val/train gap)" if modality_dropout_config.scale_with_augment else "",
        )
    else:
        log.info("Modality dropout disabled")
    if dominance_control.enabled():
        log.info(
            "Dominance controller enabled: response=%s threshold=%.3f min_delta=%.6f gradient_weight=%.2f",
            dominance_control.response,
            dominance_control.dominance_threshold,
            dominance_control.min_reliance_delta,
            dominance_control.gradient_weight,
        )
    elif track_modality_dominance:
        log.info("Dominance tracking enabled in observe-only mode")

    # ── Datasets ──────────────────────────────────────────────────────────
    n_total = (args.n_partners + args.n_beholders) if args.multiclass else 10

    train_ds = MotionDataset(
        args.data_dir, "train", args.seq_len, args.stride,
        n_persons=n_total, augment=True, multiclass=args.multiclass,
        modality_dropout=modality_dropout_config,
        flow_mode=args.flow_mode, flow_dense_size=args.flow_dense_size,
    )
    val_ds = MotionDataset(
        args.data_dir, "val", args.seq_len, args.stride,
        n_persons=n_total, augment=False, multiclass=args.multiclass,
        flow_mode=args.flow_mode, flow_dense_size=args.flow_dense_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
        persistent_workers=args.num_workers > 0,
        
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" or device.type == "cuda:0"),  # allow pinned memory for CUDA even if using a specific GPU
        persistent_workers=args.num_workers > 0,
    )

    log.info("Train: %d sequences (%d batches)", len(train_ds), len(train_loader))
    log.info("Val:   %d sequences (%d batches)", len(val_ds), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model_kwargs = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "dropout": args.dropout,
        "flow_mode": args.flow_mode,
        "flow_dense_size": args.flow_dense_size,
    }
    if args.multiclass:
        model_kwargs.update({
            "n_partners": args.n_partners,
            "n_beholders": args.n_beholders,
            "n_beholder_keypoints": args.n_beholder_keypoints,
        })
    if args.use_kinematics:
        model_kwargs["use_kinematics"] = True
        model_kwargs["kin_dim"] = args.kin_dim
    if args.use_ddl:
        model_kwargs["use_ddl"] = True
    if args.use_gated_fusion:
        model_kwargs["use_gated_fusion"] = True
    if args.multiclass and not args.no_difference_pathway:
        model_kwargs["use_difference_pathway"] = True
        model_kwargs["difference_dim"] = args.difference_dim

    model = FunscriptTCN(**model_kwargs).to(device)

    # checkpoint = Path("data\\models\\checkpoints_tcn\\tcn_epoch500.pt")
    # if checkpoint.exists():
    #     log.info("Loading checkpoint from %s", checkpoint)
    #     ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    #     model.load_state_dict(ckpt["model_state_dict"])

    # Resume from checkpoint
    if args.resume is not None:
        log.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Loaded model (epoch %s, val_loss=%s)",
                 ckpt.get("epoch", "?"), f"{ckpt.get('val_loss', 0):.6f}")

    params = model.count_parameters()
    log.info("Model: %s trainable / %s total parameters",
             f"{params['trainable']:,}", f"{params['total']:,}")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        _build_optimizer_param_groups(
            model,
            base_lr=args.lr,
            weight_decay=args.weight_decay,
            pose_lr_scale=args.pose_lr_scale,
            embedding_lr_scale=args.embedding_lr_scale,
            flow_lr_scale=args.flow_lr_scale,
            aux_lr_scale=args.aux_lr_scale,
            shared_lr_scale=args.shared_lr_scale,
        ),
    )
    live_lr_scales = {
        "pose": args.pose_lr_scale,
        "embedding": args.embedding_lr_scale,
        "flow": args.flow_lr_scale,
        "aux": args.aux_lr_scale,
        "shared": args.shared_lr_scale,
    }
    log.info(
        "Optimizer groups: %s",
        ", ".join(f"{group['name']}={group['lr']:.2e}" for group in optimizer.param_groups),
    )
    if args.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
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

    _apply_live_lr_scales(optimizer, scheduler, args.lr, live_lr_scales)


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

    model_config = extract_model_config({
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "kernel_size": 3,
        "dropout": args.dropout,
        "n_keypoints": 21,
        "embed_dim": 512,
        "flow_dim": 64,
        "flow_mode": args.flow_mode,
        "flow_dense_size": args.flow_dense_size,
    })
    if args.multiclass:
        model_config.update({
            "n_partners": args.n_partners,
            "n_beholders": args.n_beholders,
            "n_beholder_keypoints": args.n_beholder_keypoints,
        })
    else:
        model_config["n_persons"] = 10
    if args.use_kinematics:
        model_config["use_kinematics"] = True
        model_config["kin_dim"] = args.kin_dim
    if args.use_ddl:
        model_config["use_ddl"] = True
    if args.use_gated_fusion:
        model_config["use_gated_fusion"] = True
    if args.multiclass and not args.no_difference_pathway:
        model_config["use_difference_pathway"] = True
        model_config["difference_dim"] = args.difference_dim

    data_config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "stats_path": str(train_ds.stats_path) if train_ds.stats_path is not None else None,
    }
    metric_config = {
        "event_weight": loss_config.event_weight,
        "event_activity_gain": loss_config.event_activity_gain,
        "event_activity_power": loss_config.event_activity_power,
        "active_quantile": loss_config.active_quantile,
        "temporal_weight": loss_config.temporal_weight,
        "velocity_weight": loss_config.velocity_weight,
        "spectral_weight": loss_config.spectral_weight,
        "spectral_kernel": loss_config.spectral_kernel,
    }
    optimization_config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "pose_lr_scale": args.pose_lr_scale,
        "embedding_lr_scale": args.embedding_lr_scale,
        "flow_lr_scale": args.flow_lr_scale,
        "aux_lr_scale": args.aux_lr_scale,
        "shared_lr_scale": args.shared_lr_scale,
        "modality_dropout": {
            "pose_prob": modality_dropout_config.pose_prob,
            "embedding_prob": modality_dropout_config.embedding_prob,
            "flow_prob": modality_dropout_config.flow_prob,
            "scale_with_augment": modality_dropout_config.scale_with_augment,
        },
        "track_modality_dominance": track_modality_dominance,
        "dominance_eval_every": args.dominance_eval_every,
        "dominance_max_batches": args.dominance_max_batches,
        "dominance_response": dominance_control.response,
        "dominance_threshold": dominance_control.dominance_threshold,
        "dominance_min_delta": dominance_control.min_reliance_delta,
        "dominance_gradient_weight": dominance_control.gradient_weight,
        "dominance_lr_decay": dominance_control.lr_decay,
        "dominance_min_lr_scale": dominance_control.min_lr_scale,
        "dominance_dropout_step": dominance_control.dropout_increase,
        "dominance_max_dropout": dominance_control.max_dropout_prob,
    }
    last_dominance_action: dict[str, object] | None = None
    _refresh_optimization_config(optimization_config, live_lr_scales, train_ds.modality_dropout, last_dominance_action)

    log.info("Run dir: %s", run_dir)
    log.info("Checkpoint dir: %s", checkpoint_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    _early_stop_counter = 0




    best_val_loss = float("inf")
    if args.resume is not None:
        if "val_loss" in ckpt and args.load_best_val_loss:
            best_val_loss = ckpt["val_loss"]
            log.info("Resuming with best_val_loss = %.6f", best_val_loss)
        else:
            log.info("No val_loss found in checkpoint — running initial validation to get baseline val loss for early stopping")
            avg_val, avg_val_metrics, avg_pred_mean, avg_pred_std = _run_validation_epoch(
                model,
                val_loader,
                device,
                loss_config,
            )
            log.info("Initial validation loss: %.6f", avg_val)
            log.info("Initial validation metrics: %s", ", ".join(f"{k}={v:.6f}" for k, v in avg_val_metrics.items()))
            log.info("Initial validation prediction mean: %.6f, std: %.6f", avg_pred_mean, avg_pred_std)
            best_val_loss = avg_val












    original_best_val_loss = best_val_loss
    improved = False

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        train_metric_history = _init_metric_history()
        grad_share_history = {str(group.get("name", "group")): [] for group in optimizer.param_groups}
        grad_norm_history = {str(group.get("name", "group")): [] for group in optimizer.param_groups}

        for batch in train_loader:
            kp = batch["keypoints"].to(device, non_blocking=True)
            emb = batch["embeddings"].to(device, non_blocking=True)
            fl = batch["flow"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred, loss, metric_batch, loss_terms = _compute_batch_loss(
                    model,
                    kp,
                    emb,
                    fl,
                    lbl,
                    loss_config,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if track_modality_dominance:
                batch_grad_norms, batch_grad_shares = _collect_gradient_group_stats(optimizer)
                for name, value in batch_grad_norms.items():
                    grad_norm_history[name].append(value)
                for name, value in batch_grad_shares.items():
                    grad_share_history[name].append(value)
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
                writer.add_scalar("train/pos_loss", loss_terms["pos_loss"].item(), global_step)
                writer.add_scalar("train/pos_mse", metric_batch["pos_mse"].item(), global_step)
                writer.add_scalar("train/event_mse", metric_batch["event_mse"].item(), global_step)
                writer.add_scalar("train/active_mse", metric_batch["active_mse"].item(), global_step)
                writer.add_scalar("train/vel_mae", metric_batch["vel_mae"].item(), global_step)
                writer.add_scalar("train/acc_mae", metric_batch["acc_mae"].item(), global_step)
                writer.add_scalar("train/temp_loss", loss_terms["temp_loss"].item(), global_step)
                writer.add_scalar("train/vel_loss", loss_terms["vel_loss"].item(), global_step)
                writer.add_scalar("train/spec_loss", loss_terms["spec_loss"].item(), global_step)
                writer.add_scalar("train/pred_mean", pred.mean().item(), global_step)
                writer.add_scalar("train/pred_std", pred.std().item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                for param_group in optimizer.param_groups:
                    writer.add_scalar(f"lr/{param_group['name']}", param_group["lr"], global_step)

        avg_train = np.mean(train_losses)
        avg_train_metrics = {key: float(np.mean(values)) for key, values in train_metric_history.items()}

        # --- Validate ---
        avg_val, avg_val_metrics, avg_pred_mean, avg_pred_std = _run_validation_epoch(
            model,
            val_loader,
            device,
            loss_config,
        )

        epoch_time = time.time() - epoch_start

        log.info(
            "Epoch %3d/%d | train=%.6f val=%.6f pos=%.6f event=%.6f active=%.6f vel_mae=%.6f | "
            "pred_μ=%.3f pred_σ=%.3f | lr=%.2e | %.1fs",
            epoch, args.epochs, avg_train, avg_val, avg_val_metrics["pos_mse"],
            avg_val_metrics["event_mse"], avg_val_metrics["active_mse"], avg_val_metrics["vel_mae"],
            avg_pred_mean, avg_pred_std,
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        writer.add_scalar("val/loss", avg_val, epoch)
        writer.add_scalar("val/pos_loss", avg_val_metrics["pos_mse"], epoch)
        writer.add_scalar("val/event_mse", avg_val_metrics["event_mse"], epoch)
        writer.add_scalar("val/active_mse", avg_val_metrics["active_mse"], epoch)
        writer.add_scalar("val/vel_mae", avg_val_metrics["vel_mae"], epoch)
        writer.add_scalar("val/acc_mae", avg_val_metrics["acc_mae"], epoch)
        writer.add_scalar("val/pred_mean", avg_pred_mean, epoch)
        writer.add_scalar("val/pred_std", avg_pred_std, epoch)
        writer.add_scalar("train/loss_epoch", avg_train, epoch)
        writer.add_scalar("train/pos_loss_epoch", avg_train_metrics["pos_mse"], epoch)
        writer.add_scalar("train/event_mse_epoch", avg_train_metrics["event_mse"], epoch)
        writer.add_scalar("train/active_mse_epoch", avg_train_metrics["active_mse"], epoch)

        if modality_dropout_config.scale_with_augment:
            augment_scale = _compute_augment_scale(avg_val, avg_train)
            train_ds.augment_scale = augment_scale
            writer.add_scalar("train/modality_dropout_scale", augment_scale, epoch)
            log.info("  Modality dropout scale: %.3f (val/train ratio=%.3f)", augment_scale, avg_val / max(avg_train, 1e-9))
        else:
            train_ds.augment_scale = 0.0

        if track_modality_dominance:
            avg_grad_shares = {
                name: float(np.mean(values)) if values else 0.0
                for name, values in grad_share_history.items()
            }
            avg_grad_norms = {
                name: float(np.mean(values)) if values else 0.0
                for name, values in grad_norm_history.items()
            }
            for name, value in avg_grad_shares.items():
                writer.add_scalar(f"train/grad_share/{name}", value, epoch)
            for name, value in avg_grad_norms.items():
                writer.add_scalar(f"train/grad_norm_group/{name}", value, epoch)
            log.info(
                "  Gradient share: %s",
                ", ".join(f"{name}={value:.3f}" for name, value in avg_grad_shares.items()),
            )

            if epoch % args.dominance_eval_every == 0:
                baseline_loss, ablated_losses, reliance_delta, reliance_share = _estimate_modality_reliance(
                    model,
                    val_loader,
                    device,
                    loss_config,
                    max_batches=args.dominance_max_batches,
                )
                normalized_grad_shares = _normalize_modality_shares(avg_grad_shares)
                dominant_modality, combined_scores, selection_reason = _select_dominant_modality(
                    normalized_grad_shares,
                    reliance_delta,
                    reliance_share,
                    dominance_control,
                )
                for name, value in reliance_delta.items():
                    writer.add_scalar(f"val/modality_reliance_delta/{name}", value, epoch)
                for name, value in reliance_share.items():
                    writer.add_scalar(f"val/modality_reliance_share/{name}", value, epoch)
                for name, value in normalized_grad_shares.items():
                    writer.add_scalar(f"train/grad_share_modalities/{name}", value, epoch)
                for name, value in combined_scores.items():
                    writer.add_scalar(f"val/modality_combined_score/{name}", value, epoch)
                log.info(
                    "  Validation reliance: base=%.6f | %s | combined=%s | candidate=%s",
                    baseline_loss,
                    ", ".join(
                        f"{name}=abl{ablated_losses[name]:.6f}/Δ{reliance_delta[name]:.6f}/share{reliance_share[name]:.3f}"
                        for name in ("pose", "embedding", "flow")
                    ),
                    ", ".join(f"{name}={combined_scores[name]:.3f}" for name in MODALITY_NAMES),
                    dominant_modality if dominant_modality is not None else "none",
                )
                if dominance_control.enabled() and dominant_modality is not None:
                    last_dominance_action = _apply_dominance_response(
                        dominant_modality,
                        dominance_control,
                        train_ds,
                        optimizer,
                        scheduler,
                        args.lr,
                        live_lr_scales,
                    )
                    lr_change = last_dominance_action.get("lr_scale_change")
                    dropout_change = last_dominance_action.get("dropout_change")
                    change_parts = []
                    if lr_change is not None:
                        change_parts.append(
                            f"{dominant_modality}_lr_scale {lr_change['old']:.3f}->{lr_change['new']:.3f}"
                        )
                    if dropout_change is not None:
                        change_parts.append(
                            f"{dominant_modality}_dropout {dropout_change['old']:.3f}->{dropout_change['new']:.3f}"
                        )
                    log.info(
                        "  Dominance response applied: %s",
                        ", ".join(change_parts) if change_parts else "no-op (already at configured floor/cap)",
                    )
                elif dominance_control.enabled():
                    last_dominance_action = {
                        "mode": dominance_control.response,
                        "dominant_modality": None,
                        "reason": selection_reason,
                    }
                    log.info("  Dominance response skipped: %s", selection_reason)

        _refresh_optimization_config(optimization_config, live_lr_scales, train_ds.modality_dropout, last_dominance_action)
        for modality_name in MODALITY_NAMES:
            writer.add_scalar(f"control/live_lr_scale/{modality_name}", live_lr_scales[modality_name], epoch)
            writer.add_scalar(
                f"control/live_dropout/{modality_name}",
                getattr(train_ds.modality_dropout, _dropout_field_name(modality_name)),
                epoch,
            )

        if args.scheduler == "ReduceLROnPlateau":
            scheduler.step(avg_val)

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
                "data_config": data_config,
                "metric_config": metric_config,
                "optimization_config": optimization_config,
            }, checkpoint_dir / "best_tcn.pt")
            log.info("  → New best val loss: %.6f", avg_val)
        else:
            _early_stop_counter += 1
            if args.early_stopping_patience > 0 and _early_stop_counter >= args.early_stopping_patience:
                log.info(
                    "Early stopping: no improvement for %d epochs (best val=%.6f)",
                    _early_stop_counter, best_val_loss,
                )
                _early_stop_counter = 0
                # reusing this for model blending with best val model, since we want to continue training after early stopping anyway
                best_model_path = checkpoint_dir / "best_tcn.pt"
                best_model = FunscriptTCN(**model_config).to(device)
                if best_model_path.exists():
                    log.info("Loading best model from %s for continued training", best_model_path)
                    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
                    best_model.load_state_dict(ckpt["model_state_dict"])
                    # blend weights onto the current model (simple moving average with blending factor)
                    blending_factor = 0.5
                    with torch.no_grad():
                        for p, best_p in zip(model.parameters(), best_model.parameters()):
                            p.data = blending_factor * p.data + (1 - blending_factor) * best_p.data
                            

                else:
                    log.warning("Best model checkpoint not found at %s — cannot load for continued training", best_model_path)
                    break
                

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "model_config": model_config,
                "global_step": global_step,
                "data_config": data_config,
                "metric_config": metric_config,
                "optimization_config": optimization_config,
            }, checkpoint_dir / f"tcn_epoch{epoch}.pt")

        if best_val_loss < original_best_val_loss:
            improved = True
        # if epoch == 5 and not improved:
        #     break

    writer.close()
    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    train()
