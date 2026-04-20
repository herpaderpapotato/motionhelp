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
}


def extract_disposition_config(config: dict[str, object]) -> dict[str, object]:
    """Filter checkpoint config down to DispositionTCN constructor kwargs."""
    return {key: config[key] for key in DISPOSITION_CONFIG_KEYS if key in config}


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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.roi_size = roi_size
        self.d_model = d_model
        self.n_persons = n_persons
        self.encoder_dim = encoder_dim

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

        x = x.transpose(1, 2)  # [B, d_model, T]
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # [B, T, d_model]

        out = self.output_head(x).squeeze(-1)  # [B, T]
        return torch.sigmoid(out)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
