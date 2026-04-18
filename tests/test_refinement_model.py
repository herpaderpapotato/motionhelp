import torch

from postprocessing.src.models.refinement import RefinementTCN


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