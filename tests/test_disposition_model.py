import torch

from src.models.dispositiontcn import DispositionTCN, extract_disposition_config


def test_extract_disposition_config_keeps_aux_metadata() -> None:
    config = {
        "in_channels": 1280,
        "d_model": 128,
        "n_blocks": 3,
        "encoder_dim": 64,
        "use_aux_layers": True,
        "scale_channel_slices": {"p3": [0, 256], "p4": [256, 768], "p5": [768, 1280]},
        "scale_names": ["p3", "p4", "p5"],
        "seq_len": 120,
    }

    filtered = extract_disposition_config(config)

    assert filtered == {
        "in_channels": 1280,
        "d_model": 128,
        "n_blocks": 3,
        "encoder_dim": 64,
        "use_aux_layers": True,
        "scale_channel_slices": {"p3": [0, 256], "p4": [256, 768], "p5": [768, 1280]},
        "scale_names": ["p3", "p4", "p5"],
    }


def test_disposition_tcn_aux_forward_shape() -> None:
    model = DispositionTCN(
        in_channels=12,
        d_model=32,
        n_blocks=2,
        dropout=0.0,
        encoder_dim=16,
        use_aux_layers=True,
        scale_channel_slices={"p3": [0, 4], "p4": [4, 8], "p5": [8, 12]},
        scale_names=("p3", "p4", "p5"),
    )
    model.eval()

    spatial = torch.rand(2, 10, 1, 12, 7, 7)
    conf = torch.rand(2, 10, 1)

    with torch.no_grad():
        out = model(spatial, conf)

    assert out.shape == (2, 4, 10)
    assert model.aux_scale_names == ("p3", "p4", "p5")
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_disposition_tcn_disable_aux_layers_restores_fused_output_shape() -> None:
    model = DispositionTCN(
        in_channels=12,
        d_model=32,
        n_blocks=2,
        dropout=0.0,
        encoder_dim=16,
        use_aux_layers=True,
        scale_channel_slices={"p3": [0, 4], "p4": [4, 8], "p5": [8, 12]},
        scale_names=("p3", "p4", "p5"),
    )
    model.disable_aux_layers()
    model.eval()

    spatial = torch.rand(1, 8, 1, 12, 7, 7)
    conf = torch.rand(1, 8, 1)

    with torch.no_grad():
        out = model(spatial, conf)

    assert out.shape == (1, 8)