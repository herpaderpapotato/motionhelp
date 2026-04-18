"""Refinement model for postprocessing TCN predictions.

A small 1D convolutional network with dilated convolutions that takes
per-frame predictions from the upstream TCN and refines them to better
match ground-truth labels over long sequences (1200 frames).
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

    Takes single-channel per-frame predictions [B, T] and outputs
    refined predictions [B, T] in [0, 1].

    Architecture:
    - Input projection: 1 → channels
    - Stack of dilated residual blocks with exponentially growing dilation
      (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) for ~2047-frame receptive field
    - Output projection: channels → 1 → sigmoid
    - Global residual: output = sigmoid(learned_refinement + alpha * input)
    """

    def __init__(
        self,
        channels: int = 64,
        n_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks

        # Input: map raw prediction to feature space
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

        # Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] raw predictions in [0, 1]
        Returns:
            [B, T] refined predictions in [0, 1]
        """
        # Transform input to logit space for better learning
        x_logit = torch.logit(x.clamp(1e-4, 1 - 1e-4))

        h = x_logit.unsqueeze(1)  # [B, 1, T]
        h = self.input_proj(h)    # [B, C, T]

        for block in self.blocks:
            h = block(h)

        refined_logit = self.output_proj(h).squeeze(1)  # [B, T]

        # Residual in logit space
        out = torch.sigmoid(refined_logit + self.alpha * x_logit)
        return out
