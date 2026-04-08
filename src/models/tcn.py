"""Temporal Convolutional Network for funscript prediction.

Takes per-frame video features (pose keypoints, YOLO embeddings, optical flow)
and predicts per-frame funscript position values in [0, 1].

Architecture:
    1. Person-aware feature encoding with attention pooling
    2. Feature fusion (pose + embeddings + flow)
    3. Dilated TCN backbone for multi-scale temporal modeling
    4. Per-frame output head with sigmoid

Design rationale vs. previous Transformer approach:
    - Dilated convolutions capture temporal structure without attention
      homogenization that caused mean-collapse in the Transformer
    - Person attention pools across detections weighted by confidence,
      avoiding the 10-person flattening that dominated input dimensionality
    - Bidirectional (non-causal) since we process full clips
    - Receptive field with 6 blocks, k=3: 1 + 4*(1+2+4+8+16+32) = 253 frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonAttention(nn.Module):
    """Attention pooling across person detections, biased by keypoint confidence.

    Learns which person detections are most relevant for prediction,
    using a trainable projection biased by detection confidence scores.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.score_proj = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D] per-person features
            conf: [B, T, N] per-person confidence (higher = more reliable)
        Returns: [B, T, D]
        """
        scores = self.score_proj(x).squeeze(-1)  # [B, T, N]
        if conf is not None:
            scores = scores + conf.clamp(min=1e-6).log()
        weights = F.softmax(scores, dim=-1)  # [B, T, N]
        return (x * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]


class TCNBlock(nn.Module):
    """Residual block with bidirectional dilated convolutions.

    Two dilated conv layers with BatchNorm, GELU activation, and dropout.
    Uses symmetric (same) padding for bidirectional processing.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # same padding
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → [B, C, T]"""
        return self.act(x + self.net(x))


class FunscriptTCN(nn.Module):
    """Temporal convolutional network for per-frame funscript prediction.

    Input features (per frame):
        keypoints:  [B, T, N_persons, N_keypoints, 3]  (x, y, confidence)
        embeddings: [B, T, N_persons, embed_dim]
        flow:       [B, T, flow_dim]

    Output: [B, T] position values in [0, 1]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_persons: int = 10,
        n_keypoints: int = 21,
        embed_dim: int = 512,
        flow_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_persons = n_persons
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim

        kp_feat_dim = n_keypoints * 3  # 63

        # -- Per-person feature encoders --
        self.pose_encoder = nn.Sequential(
            nn.Linear(kp_feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.pose_attn = PersonAttention(128)

        self.emb_encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.emb_attn = PersonAttention(128)

        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # -- Feature fusion: 128 (pose) + 128 (emb) + 64 (flow) = 320 --
        fusion_in = 128 + 128 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # -- TCN backbone --
        dilations = [2 ** i for i in range(n_blocks)]  # [1, 2, 4, 8, 16, 32]
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, d, dropout) for d in dilations
        ])

        # -- Output head --
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        embeddings: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            keypoints:  [B, T, N, K, 3]
            embeddings: [B, T, N, E]
            flow:       [B, T, F]
        Returns: [B, T] positions in [0, 1]
        """
        B, T = keypoints.shape[:2]

        # -- Pose --
        kp_flat = keypoints.reshape(B, T, self.n_persons, -1)     # [B, T, N, K*3]
        kp_conf = keypoints[:, :, :, :, 2].mean(dim=-1)           # [B, T, N]
        pose_feat = self.pose_encoder(kp_flat)                     # [B, T, N, 128]
        pose_out = self.pose_attn(pose_feat, kp_conf)              # [B, T, 128]

        # -- Embeddings --
        emb_feat = self.emb_encoder(embeddings)                    # [B, T, N, 128]
        emb_out = self.emb_attn(emb_feat, kp_conf)                # [B, T, 128]

        # -- Flow --
        flow_out = self.flow_encoder(flow)                         # [B, T, 64]

        # -- Fusion --
        fused = torch.cat([pose_out, emb_out, flow_out], dim=-1)   # [B, T, 320]
        x = self.fusion(fused)                                     # [B, T, d_model]

        # -- TCN (expects channels-first) --
        x = x.transpose(1, 2)                                     # [B, d_model, T]
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)                                     # [B, T, d_model]

        # -- Output --
        out = self.output_head(x).squeeze(-1)                      # [B, T]
        return torch.sigmoid(out)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
