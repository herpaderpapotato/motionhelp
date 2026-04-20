import torch

from postprocessing.src.models.refinement import (
    CalibrationRefiner,
    ChunkedRefinementAutoencoder,
    CurveRefinementUNet,
    MotionPriorRefiner,
    RefinementAutoencoder,
    RefinementTCN,
    build_refinement_model,
)


def test_value_mode_starts_as_identity() -> None:
    model = RefinementTCN(channels=16, n_blocks=2, residual_mode="value")
    model.eval()

    x = torch.rand(2, 64)
    with torch.no_grad():
        out = model(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_value_mode_uses_bounded_position_corrections() -> None:
    model = RefinementTCN(
        channels=8,
        n_blocks=1,
        residual_mode="value",
        delta_limit=0.25,
    )
    model.eval()

    with torch.no_grad():
        model.output_proj[-1].bias.fill_(10.0)

    x = torch.full((1, 32), 0.2)
    with torch.no_grad():
        out = model(x)

    assert torch.all(out >= x)
    assert torch.all(out <= 0.45 + 1e-6)


def test_curve_unet_starts_as_identity() -> None:
    model = CurveRefinementUNet(channels=32, n_blocks=2)
    model.eval()

    x = torch.rand(2, 96)
    with torch.no_grad():
        out = model(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_build_refinement_model_supports_curve_unet() -> None:
    model = build_refinement_model(model_type="curve-unet", channels=32, n_blocks=2)
    assert isinstance(model, CurveRefinementUNet)


def test_autoencoder_refiner_preserves_shape_and_range() -> None:
    model = RefinementAutoencoder(seq_len=96, latent_dim=16, depth=3, encoder_style="dense2")
    model.eval()

    x = torch.rand(2, 96)
    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_build_refinement_model_supports_autoencoder() -> None:
    model = build_refinement_model(model_type="autoencoder", seq_len=96, channels=32, latent_dim=16)
    assert isinstance(model, ChunkedRefinementAutoencoder)


def test_chunked_autoencoder_handles_longer_sequences() -> None:
    model = ChunkedRefinementAutoencoder(chunk_len=32, chunk_stride=16, latent_dim=8, depth=2)
    model.eval()

    x = torch.rand(2, 96)
    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_motion_prior_refiner_starts_from_prior_projection() -> None:
    model = MotionPriorRefiner(
        seq_len=96,
        channels=16,
        n_blocks=2,
        latent_dim=8,
        ae_depth=2,
        ae_style="pool",
    )
    model.eval()

    x = torch.rand(2, 96)
    with torch.no_grad():
        expected = model.prior.decode(model.prior.encode(x))
        out = model(x)

    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_calibration_refiner_starts_as_identity() -> None:
    model = CalibrationRefiner(channels=32, kernel_size=5, n_knots=8)
    model.eval()

    x = torch.rand(2, 96)
    with torch.no_grad():
        out = model(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_build_refinement_model_supports_calibration() -> None:
    model = build_refinement_model(model_type="calibration", channels=32, kernel_size=5)
    assert isinstance(model, CalibrationRefiner)


def test_build_refinement_model_supports_prior_refiner() -> None:
    model = build_refinement_model(
        model_type="prior-refiner",
        seq_len=96,
        channels=16,
        latent_dim=8,
        ae_depth=2,
        ae_style="pool",
    )
    assert isinstance(model, MotionPriorRefiner)