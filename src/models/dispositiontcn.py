"""DispositionTCN: Simplified TCN using raw spatial RoI features.

Replaces the multi-stream (pose + embedding + flow) approach with a single
SpatialDeltaEncoder that ingests RoI-aligned YOLO spatial grids [B, T, N, C, H, W]
and computes both structure and motion features natively via temporal deltas.

Architecture:
    1. SpatialDeltaEncoder: fuses frame structure (X) with motion (delta X)
       via concatenation and 2D CNN spatial reduction -> [B, T, N, encoder_dim]
    2. PersonAttention: confidence-weighted pooling across N persons -> [B, T, encoder_dim]
    3. Linear projection -> [B, T, d_model]
    4. Dilated TCN backbone for multi-scale temporal modeling
    5. Per-frame output head with sigmoid -> [B, T] in [0, 1]

Data flow:
    Input: (spatial_features [B, T, N, C, H, W], conf [B, T, N])
    -> SpatialDeltaEncoder -> PersonAttention -> projection -> TCN -> output head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from src.models.tcn import PersonAttention, TCNBlock, DualDilatedBlock


DISPOSITION_CONFIG_KEYS = {
    "in_channels",
    "roi_size",
    "d_model",
    "n_blocks",
    "kernel_size",
    "dropout",
    "n_persons",
    "encoder_dim",
    "use_ddl",
    "use_aux_layers",
    "scale_channel_slices",
    "scale_names",
}


def extract_disposition_config(config: dict[str, object]) -> dict[str, object]:
    """Filter checkpoint config down to DispositionTCN constructor kwargs."""
    return {key: config[key] for key in DISPOSITION_CONFIG_KEYS if key in config}


def _normalize_scale_channel_slices(
    scale_channel_slices: dict[str, Sequence[int]] | Sequence[Sequence[int]] | None,
    scale_names: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Normalise persisted scale slice metadata for aux branch construction."""
    if scale_channel_slices is None:
        return (), ()

    if isinstance(scale_channel_slices, dict):
        if scale_names is None:
            names = tuple(str(name) for name in scale_channel_slices.keys())
        else:
            names = tuple(str(name) for name in scale_names)
            missing = [name for name in names if name not in scale_channel_slices]
            if missing:
                raise ValueError(
                    "scale_channel_slices is missing entries for " + ", ".join(missing)
                )
        raw_slices = [scale_channel_slices[name] for name in names]
    else:
        raw_slices = list(scale_channel_slices)
        if scale_names is None:
            names = tuple(f"scale_{idx}" for idx in range(len(raw_slices)))
        else:
            names = tuple(str(name) for name in scale_names)
        if len(names) != len(raw_slices):
            raise ValueError("scale_names and scale_channel_slices must have the same length")

    normalized: list[tuple[int, int]] = []
    previous_end = -1
    for raw_slice in raw_slices:
        if len(raw_slice) != 2:
            raise ValueError("Each scale channel slice must contain exactly [start, end]")
        start, end = int(raw_slice[0]), int(raw_slice[1])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid scale channel slice [{start}, {end}]")
        if start < previous_end:
            raise ValueError("Scale channel slices must be ordered and non-overlapping")
        normalized.append((start, end))
        previous_end = end

    return names, tuple(normalized)


class SpatialDeltaEncoder(nn.Module):
    """Replaces Pose, Flow, and 1D Embedding encoders.

    Ingests raw RoI-aligned YOLO spatial grids and computes temporal deltas
    to capture both structure and motion in a single unified representation.
    """

    def __init__(self, in_channels: int, out_dim: int = 128):
        super().__init__()
        fused_channels = in_channels * 2  # concatenate frame + delta

        # 2D CNN to reduce spatial grid to a 1D token
        # 14x14 -> 7x7 -> 3x3 -> 1x1 (via adaptive pool)
        self.spatial_net = nn.Sequential(
            nn.Conv2d(fused_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C, H, W] - RoI-aligned spatial features from YOLO
        Returns: [B, T, N, out_dim]
        """
        B, T, N, C, H, W = x.shape

        # Compute latent deltas (temporal differences replace optical flow)
        x_prev = torch.cat([x[:, :1, ...], x[:, :-1, ...]], dim=1)
        delta_x = x - x_prev  # [B, T, N, C, H, W]

        # Fuse structure + motion
        fused = torch.cat([x, delta_x], dim=3)  # [B, T, N, C*2, H, W]

        # Spatial reduction via 2D CNN
        fused_flat = fused.view(B * T * N, C * 2, H, W)
        encoded_flat = self.spatial_net(fused_flat)  # [B*T*N, out_dim]

        return encoded_flat.view(B, T, N, -1)  # [B, T, N, out_dim]


class DispositionTCN(nn.Module):
    """Temporal convolutional network using spatial RoI features.

    Single-input architecture that replaces separate pose, embedding, and flow
    encoders with a unified SpatialDeltaEncoder operating on raw YOLO RoI grids.

    Input:
        spatial_features: [B, T, N, C, H, W]  (RoI-aligned backbone features)
        conf:             [B, T, N]            (detection confidence scores)

    Output: [B, T] position values in [0, 1]
        When use_aux_layers=True, returns [B, 1 + n_scales, T] where channel 0 is
        the fused prediction and the remaining channels are per-scale aux outputs.
    """

    def __init__(
        self,
        in_channels: int = 512,
        roi_size: int = 7,
        d_model: int = 256,
        n_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_persons: int = 1,
        encoder_dim: int = 128,
        use_ddl: bool = False,
        use_aux_layers: bool = False,
        scale_channel_slices: dict[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
        scale_names: Sequence[str] | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.roi_size = roi_size
        self.d_model = d_model
        self.n_persons = n_persons
        self.encoder_dim = encoder_dim
        self.use_aux_layers = use_aux_layers
        self.aux_scale_names, self.scale_channel_slices = _normalize_scale_channel_slices(
            scale_channel_slices,
            scale_names=scale_names,
        )

        if self.scale_channel_slices and self.scale_channel_slices[-1][1] > in_channels:
            raise ValueError(
                "scale_channel_slices extends past in_channels "
                f"({self.scale_channel_slices[-1][1]} > {in_channels})"
            )
        if self.use_aux_layers:
            if not self.scale_channel_slices:
                raise ValueError(
                    "use_aux_layers=True requires scale_channel_slices metadata from the spatial cache"
                )
            if self.scale_channel_slices[0][0] != 0 or self.scale_channel_slices[-1][1] != in_channels:
                raise ValueError(
                    "scale_channel_slices must span the full concatenated input when aux layers are enabled"
                )

        self.spatial_encoder = SpatialDeltaEncoder(in_channels, out_dim=encoder_dim)
        self.person_attn = PersonAttention(encoder_dim)

        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        dilations = [2 ** i for i in range(n_blocks)]
        block_cls = DualDilatedBlock if use_ddl else TCNBlock
        self.tcn_blocks = nn.ModuleList([
            block_cls(d_model, kernel_size, d, dropout) for d in dilations
        ])

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        if self.use_aux_layers:
            aux_dilations = [2 ** i for i in range(min(3, n_blocks))]
            self.aux_spatial_encoders = nn.ModuleList()
            self.aux_person_attn = nn.ModuleList()
            self.aux_proj = nn.ModuleList()
            self.aux_tcn_blocks = nn.ModuleList()
            self.aux_output_heads = nn.ModuleList()

            for start, end in self.scale_channel_slices:
                branch_channels = end - start
                self.aux_spatial_encoders.append(
                    SpatialDeltaEncoder(branch_channels, out_dim=encoder_dim)
                )
                self.aux_person_attn.append(PersonAttention(encoder_dim))
                self.aux_proj.append(
                    nn.Sequential(
                        nn.Linear(encoder_dim, d_model),
                        nn.LayerNorm(d_model),
                        nn.GELU(),
                    )
                )
                self.aux_tcn_blocks.append(
                    nn.ModuleList([
                        block_cls(d_model, kernel_size, d, dropout) for d in aux_dilations
                    ])
                )
                self.aux_output_heads.append(
                    nn.Sequential(
                        nn.Linear(d_model, 64),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 1),
                    )
                )

    @staticmethod
    def _run_tcn(x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C, T]
        for block in blocks:
            x = block(x)
        return x.transpose(1, 2)  # [B, T, C]

    def disable_aux_layers(self) -> None:
        """Permanently disable aux branches after loading a checkpoint."""
        self.use_aux_layers = False
        self.aux_scale_names = ()
        self.scale_channel_slices = ()
        for attr in (
            "aux_spatial_encoders",
            "aux_person_attn",
            "aux_proj",
            "aux_tcn_blocks",
            "aux_output_heads",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def forward(
        self,
        spatial_features: torch.Tensor,
        conf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: [B, T, N, C, H, W] RoI-aligned spatial features
            conf: [B, T, N] per-person detection confidence (optional)
        Returns: [B, T] positions in [0, 1]
        """
        encoded = self.spatial_encoder(spatial_features)  # [B, T, N, encoder_dim]
        pooled = self.person_attn(encoded, conf)  # [B, T, encoder_dim]
        x = self.proj(pooled)  # [B, T, d_model]

        aux_outputs: list[torch.Tensor] = []
        if self.use_aux_layers:
            for idx, (start, end) in enumerate(self.scale_channel_slices):
                scale_input = spatial_features[:, :, :, start:end, :, :]
                scale_encoded = self.aux_spatial_encoders[idx](scale_input)
                scale_pooled = self.aux_person_attn[idx](scale_encoded, conf)
                scale_x = self.aux_proj[idx](scale_pooled)
                scale_x = self._run_tcn(scale_x, self.aux_tcn_blocks[idx])
                aux_outputs.append(self.aux_output_heads[idx](scale_x).squeeze(-1))

        x = self._run_tcn(x, self.tcn_blocks)

        main_out = self.output_head(x).squeeze(-1)  # [B, T]
        if self.use_aux_layers:
            out = torch.stack([main_out, *aux_outputs], dim=1)
            return torch.sigmoid(out)
        return torch.sigmoid(main_out)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
