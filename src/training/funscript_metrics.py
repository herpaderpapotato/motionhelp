"""Shared loss helpers and evaluation metrics for funscript regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _first_difference(x: torch.Tensor) -> torch.Tensor:
    """Return first-order temporal difference with zero padding at the start."""
    diff = torch.zeros_like(x)
    if x.shape[1] > 1:
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
    return diff


def _second_difference(x: torch.Tensor) -> torch.Tensor:
    """Return second-order temporal difference with zero padding at the start."""
    diff = torch.zeros_like(x)
    if x.shape[1] > 2:
        diff[:, 2:] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return diff


def high_frequency_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Compute MSE on high-frequency residuals after moving-average smoothing."""
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, device=pred.device, dtype=pred.dtype) / kernel_size
    pred_3d = pred.unsqueeze(1)
    tgt_3d = target.unsqueeze(1)
    pred_smooth = F.conv1d(F.pad(pred_3d, (pad, pad), mode="replicate"), weight).squeeze(1)
    tgt_smooth = F.conv1d(F.pad(tgt_3d, (pad, pad), mode="replicate"), weight).squeeze(1)
    return F.mse_loss(pred - pred_smooth, target - tgt_smooth)


def compute_activity(
    target: torch.Tensor,
    activity_gain: float = 1.0,
    activity_power: float = 1.0,
) -> torch.Tensor:
    """Estimate frame-wise motion saliency from target derivatives."""
    velocity = _first_difference(target).abs()
    acceleration = _second_difference(target).abs()
    activity = velocity + 0.5 * acceleration
    if activity_power != 1.0:
        activity = activity.pow(activity_power)
    if activity_gain != 1.0:
        activity = activity_gain * activity
    return activity


def compute_event_weights(
    target: torch.Tensor,
    activity_gain: float = 3.0,
    activity_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized per-frame weights that emphasize active motion frames."""
    activity = compute_activity(target, activity_gain=activity_gain, activity_power=activity_power)
    weights = 1.0 + activity
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return weights, activity


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def compute_regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    spectral_kernel: int,
    activity_gain: float = 3.0,
    activity_power: float = 1.0,
    active_quantile: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Compute a consistent set of regression losses and diagnostics."""
    error = pred - target
    sq_error = error.square()
    abs_error = error.abs()

    weights, activity = compute_event_weights(
        target,
        activity_gain=activity_gain,
        activity_power=activity_power,
    )
    weighted_mse = (sq_error * weights).mean()

    velocity_pred = _first_difference(pred)
    velocity_tgt = _first_difference(target)
    velocity_error = velocity_pred - velocity_tgt
    velocity_sq_error = velocity_error.square()
    velocity_abs_error = velocity_error.abs()

    acceleration_pred = _second_difference(pred)
    acceleration_tgt = _second_difference(target)
    acceleration_error = acceleration_pred - acceleration_tgt
    acceleration_sq_error = acceleration_error.square()
    acceleration_abs_error = acceleration_error.abs()

    if target.shape[1] > 0:
        threshold = torch.quantile(activity, active_quantile, dim=1, keepdim=True)
        active_mask = activity >= threshold
    else:
        active_mask = torch.ones_like(target, dtype=torch.bool)

    metrics = {
        "pos_mse": sq_error.mean(),
        "pos_mae": abs_error.mean(),
        "event_mse": weighted_mse,
        "active_mse": _masked_mean(sq_error, active_mask),
        "active_mae": _masked_mean(abs_error, active_mask),
        "vel_mse": velocity_sq_error.mean(),
        "vel_mae": velocity_abs_error.mean(),
        "acc_mse": acceleration_sq_error.mean(),
        "acc_mae": acceleration_abs_error.mean(),
        "spec_mse": high_frequency_mse(pred, target, spectral_kernel),
        "activity_mean": activity.mean(),
    }
    return metrics