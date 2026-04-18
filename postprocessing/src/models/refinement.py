"""Refinement model for postprocessing TCN predictions.

A small 1D convolutional network with dilated convolutions that takes
per-frame funscript position predictions from the upstream TCN and refines
them to better match ground-truth labels over long sequences.
"""

import torch
import torch.nn as nn


class DilatedResBlock(nn.Module):
    """Dilated causal 1D conv block with residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # symmetric padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)
        return self.act(out + residual)


class RefinementTCN(nn.Module):
    """Postprocessing refinement network.

        Takes single-channel per-frame position predictions [B, T] and outputs
        refined positions [B, T] in [0, 1].

    Architecture:
    - Input projection: 1 → channels
    - Stack of dilated residual blocks with exponentially growing dilation
            (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        - Output projection: channels → 1 correction signal
        - Default residual path operates directly in value space to match the
            continuous [0, 1] funscript position target.

        Legacy checkpoints used a logit-space residual. That path remains
        available via residual_mode="logit" for backward compatibility.
    """

    def __init__(
        self,
        channels: int = 64,
        n_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1,
        residual_mode: str = "value",
        delta_limit: float = 0.35,
    ):
        super().__init__()
        if residual_mode not in {"value", "logit"}:
            raise ValueError("residual_mode must be 'value' or 'logit'")
        if delta_limit <= 0:
            raise ValueError("delta_limit must be positive")

        self.channels = channels
        self.n_blocks = n_blocks
        self.residual_mode = residual_mode
        self.delta_limit = delta_limit

        # Input: map the scalar position signal to feature space.
        self.input_proj = nn.Conv1d(1, channels, 1)

        # Dilated conv stack
        self.blocks = nn.ModuleList([
            DilatedResBlock(channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Output head
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels // 2, 1, 1),
        )

        if self.residual_mode == "value":
            # Start from an exact identity mapping and learn corrections.
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable residual scale.
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] raw position predictions in [0, 1]
        Returns:
            [B, T] refined positions in [0, 1]
        """
        x = x.clamp(0.0, 1.0)

        if self.residual_mode == "logit":
            residual_base = torch.logit(x.clamp(1e-4, 1 - 1e-4))
            model_input = residual_base
        else:
            residual_base = x
            model_input = (2.0 * x) - 1.0

        h = model_input.unsqueeze(1)  # [B, 1, T]
        h = self.input_proj(h)        # [B, C, T]

        for block in self.blocks:
            h = block(h)

        correction = self.output_proj(h).squeeze(1)  # [B, T]

        if self.residual_mode == "logit":
            return torch.sigmoid(correction + self.alpha * residual_base)

        bounded_delta = self.delta_limit * torch.tanh(correction)
        return torch.clamp(residual_base + self.alpha * bounded_delta, 0.0, 1.0)
