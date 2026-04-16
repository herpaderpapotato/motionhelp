"""Tests for structured pose encoding and event-aware metrics."""

import torch

from src.models.tcn import (
    DifferenceMagnitudeEncoder,
    FunscriptTCN,
    PERFORMER_21_BONES,
    StructuredPoseEncoder,
    extract_model_config,
)
from src.training.funscript_metrics import compute_regression_metrics


def test_structured_pose_encoder_is_translation_invariant() -> None:
    encoder = StructuredPoseEncoder(
        n_keypoints=21,
        output_dim=32,
        kernel_size=3,
        dropout=0.0,
        bones=PERFORMER_21_BONES,
        root_index=17,
        root_pair=(11, 12),
    )
    encoder.eval()

    keypoints = torch.rand(2, 5, 1, 21, 3)
    keypoints[..., 2] = 1.0
    shifted = keypoints.clone()
    shifted[..., 0] += 0.2
    shifted[..., 1] -= 0.15

    with torch.no_grad():
        encoded_a, conf_a = encoder(keypoints)
        encoded_b, conf_b = encoder(shifted)

    assert torch.allclose(encoded_a, encoded_b, atol=1e-5)
    assert torch.allclose(conf_a, conf_b, atol=1e-6)


def test_event_metrics_upweight_sparse_motion_errors() -> None:
    target = torch.zeros(1, 12)
    target[:, 6] = 1.0
    pred = torch.zeros_like(target)

    metrics = compute_regression_metrics(
        pred,
        target,
        spectral_kernel=5,
        activity_gain=3.0,
        activity_power=1.0,
        active_quantile=0.8,
    )

    assert metrics["event_mse"].item() > metrics["pos_mse"].item()
    assert metrics["active_mse"].item() >= metrics["pos_mse"].item()
    assert metrics["vel_mae"].item() > 0.0


def test_difference_magnitude_encoder_is_joint_shift_invariant() -> None:
    encoder = DifferenceMagnitudeEncoder(
        n_beholder_keypoints=7,
        n_partners=5,
        n_keypoints=21,
        output_dim=16,
        kernel_size=3,
        dropout=0.0,
    )
    encoder.eval()

    partner_kp = torch.rand(2, 6, 5, 21, 3)
    beholder_kp = torch.rand(2, 6, 1, 7, 3)
    partner_kp[..., 2] = 1.0
    beholder_kp[..., 2] = 1.0

    partner_shifted = partner_kp.clone()
    beholder_shifted = beholder_kp.clone()
    partner_shifted[..., 0] += 0.12
    partner_shifted[..., 1] -= 0.08
    beholder_shifted[..., 0] += 0.12
    beholder_shifted[..., 1] -= 0.08

    with torch.no_grad():
        out_a = encoder(partner_kp, beholder_kp)
        out_b = encoder(partner_shifted, beholder_shifted)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_extract_model_config_filters_checkpoint_metadata() -> None:
    config = {
        "d_model": 256,
        "n_blocks": 6,
        "dropout": 0.1,
        "n_partners": 5,
        "use_difference_pathway": True,
        "seq_len": 120,
        "event_weight": 0.25,
        "metric_config": {"active_quantile": 0.8},
    }

    filtered = extract_model_config(config)

    assert filtered == {
        "d_model": 256,
        "n_blocks": 6,
        "dropout": 0.1,
        "n_partners": 5,
        "use_difference_pathway": True,
    }


def test_funscript_tcn_multiclass_forward_shape() -> None:
    model = FunscriptTCN(
        d_model=64,
        n_blocks=2,
        dropout=0.0,
        n_partners=2,
        n_beholders=1,
        n_beholder_keypoints=7,
        use_kinematics=True,
        use_ddl=True,
        use_gated_fusion=True,
        use_difference_pathway=True,
    )
    model.eval()

    keypoints = torch.rand(2, 16, 3, 21, 3)
    keypoints[..., 2] = 1.0
    embeddings = torch.rand(2, 16, 3, 512)
    flow = torch.rand(2, 16, 64)

    with torch.no_grad():
        out = model(keypoints, embeddings, flow)

    assert out.shape == (2, 4, 16)