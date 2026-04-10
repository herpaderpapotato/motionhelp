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


class DualDilatedBlock(nn.Module):
    """Dual Dilated Layer inspired by MS-TCN++.

    Uses two parallel dilated convolutions with different dilation rates
    (d and 2d) to capture both fine-grained and coarse temporal patterns,
    eliminating the gridding artifacts of single-dilation TCN blocks.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        # Branch 1: standard dilation
        pad1 = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad1, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)

        # Branch 2: doubled dilation for broader context
        large_d = dilation * 2
        pad2 = large_d * (kernel_size - 1) // 2
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad2, dilation=large_d)
        self.bn2 = nn.BatchNorm1d(channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → [B, C, T]"""
        out1 = self.act(self.bn1(self.conv1(x)))
        out2 = self.act(self.bn2(self.conv2(x)))
        return self.act(x + self.dropout(out1 + out2))


class FunscriptTCN(nn.Module):
    """Temporal convolutional network for per-frame funscript prediction.

    Input features (per frame):
        keypoints:  [B, T, N_total, N_keypoints, 3]  (x, y, confidence)
        embeddings: [B, T, N_total, embed_dim]
        flow:       [B, T, flow_dim]

    In multiclass mode (n_partners is set):
        N_total = n_partners + n_beholders
        Slots 0..n_partners-1 are partner detections
        Slots n_partners.. are beholder detections (keypoints truncated to n_beholder_keypoints)

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
        # Multiclass extensions
        n_partners: int | None = None,
        n_beholders: int = 1,
        n_beholder_keypoints: int = 7,
        beholder_pose_dim: int = 32,
        beholder_emb_dim: int = 32,
        # Enhanced feature options
        use_kinematics: bool = False,
        use_ddl: bool = False,
        kin_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim
        self.multiclass = n_partners is not None
        self.use_kinematics = use_kinematics
        self.use_ddl = use_ddl
        self.kin_dim = kin_dim

        if self.multiclass:
            self.n_partners = n_partners
            self.n_beholders = n_beholders
            self.n_persons = n_partners  # PersonAttention uses partner count
            self.n_beholder_keypoints = n_beholder_keypoints
            self.beholder_pose_dim = beholder_pose_dim
            self.beholder_emb_dim = beholder_emb_dim
        else:
            self.n_persons = n_persons

        kp_feat_dim = n_keypoints * 3  # 63

        # -- Per-person feature encoders (partners in multiclass, all in single) --
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

        # -- Kinematic derivative encoder (velocity + acceleration) --
        if self.use_kinematics:
            kin_feat_dim = n_keypoints * 4  # vel(K*2) + acc(K*2)
            self.kin_encoder = nn.Sequential(
                nn.Linear(kin_feat_dim, kin_dim),
                nn.LayerNorm(kin_dim),
                nn.GELU(),
            )
            self.kin_attn = PersonAttention(kin_dim)

        # -- Beholder encoders (multiclass only) --
        if self.multiclass:
            beh_kp_dim = n_beholder_keypoints * 3  # 21
            self.beholder_pose_encoder = nn.Sequential(
                nn.Linear(beh_kp_dim, beholder_pose_dim),
                nn.LayerNorm(beholder_pose_dim),
                nn.GELU(),
            )
            self.beholder_emb_encoder = nn.Sequential(
                nn.Linear(embed_dim, beholder_emb_dim),
                nn.LayerNorm(beholder_emb_dim),
                nn.GELU(),
            )
            fusion_in = 128 + 128 + 64 + beholder_pose_dim + beholder_emb_dim
            if self.use_kinematics:
                fusion_in += kin_dim
        else:
            fusion_in = 128 + 128 + 64  # 320
            if self.use_kinematics:
                fusion_in += kin_dim

        # -- Feature fusion --
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # -- TCN backbone --
        dilations = [2 ** i for i in range(n_blocks)]  # [1, 2, 4, 8, 16, 32]
        block_cls = DualDilatedBlock if self.use_ddl else TCNBlock
        self.tcn_blocks = nn.ModuleList([
            block_cls(d_model, kernel_size, d, dropout) for d in dilations
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
            keypoints:  [B, T, N, K, 3]  — N = n_partners + n_beholders (multiclass)
                                            or n_persons (single-class)
            embeddings: [B, T, N, E]
            flow:       [B, T, F]
        Returns: [B, T] positions in [0, 1]
        """
        B, T = keypoints.shape[:2]

        if self.multiclass:
            # -- Split partner and beholder --
            partner_kp = keypoints[:, :, :self.n_partners]            # [B, T, Np, K, 3]
            beholder_kp = keypoints[
                :, :,
                self.n_partners : self.n_partners + self.n_beholders,
                :self.n_beholder_keypoints,
            ]                                                          # [B, T, Nb, Kb, 3]
            partner_emb = embeddings[:, :, :self.n_partners]           # [B, T, Np, E]
            beholder_emb = embeddings[
                :, :,
                self.n_partners : self.n_partners + self.n_beholders,
            ]                                                          # [B, T, Nb, E]

            # -- Partner pose (existing path) --
            kp_flat = partner_kp.reshape(B, T, self.n_partners, -1)    # [B, T, Np, K*3]
            kp_conf = partner_kp[:, :, :, :, 2].mean(dim=-1)          # [B, T, Np]
            pose_feat = self.pose_encoder(kp_flat)                     # [B, T, Np, 128]
            pose_out = self.pose_attn(pose_feat, kp_conf)              # [B, T, 128]

            # -- Partner embeddings (existing path) --
            emb_feat = self.emb_encoder(partner_emb)                   # [B, T, Np, 128]
            emb_out = self.emb_attn(emb_feat, kp_conf)                # [B, T, 128]

            # -- Beholder pose --
            beh_kp_flat = beholder_kp.reshape(B, T, self.n_beholders, -1)  # [B, T, Nb, Kb*3]
            beh_pose_feat = self.beholder_pose_encoder(beh_kp_flat)    # [B, T, Nb, beh_pose_dim]
            beh_pose_out = beh_pose_feat.mean(dim=2)                   # [B, T, beh_pose_dim]

            # -- Beholder embeddings --
            beh_emb_feat = self.beholder_emb_encoder(beholder_emb)     # [B, T, Nb, beh_emb_dim]
            beh_emb_out = beh_emb_feat.mean(dim=2)                     # [B, T, beh_emb_dim]

            # -- Flow --
            flow_out = self.flow_encoder(flow)                         # [B, T, 64]

            # -- Kinematics (velocity + acceleration from partner keypoints) --
            if self.use_kinematics:
                kin_out = self._compute_kinematics(
                    partner_kp, self.n_partners, kp_conf,
                )                                                      # [B, T, kin_dim]

            # -- Fusion (augmented with beholder + optional kinematics) --
            fusion_parts = [pose_out, emb_out, flow_out, beh_pose_out, beh_emb_out]
            if self.use_kinematics:
                fusion_parts.append(kin_out)
            fused = torch.cat(fusion_parts, dim=-1)
        else:
            # -- Original single-class path --
            kp_flat = keypoints.reshape(B, T, self.n_persons, -1)      # [B, T, N, K*3]
            kp_conf = keypoints[:, :, :, :, 2].mean(dim=-1)           # [B, T, N]
            pose_feat = self.pose_encoder(kp_flat)                     # [B, T, N, 128]
            pose_out = self.pose_attn(pose_feat, kp_conf)              # [B, T, 128]

            emb_feat = self.emb_encoder(embeddings)                    # [B, T, N, 128]
            emb_out = self.emb_attn(emb_feat, kp_conf)                # [B, T, 128]

            flow_out = self.flow_encoder(flow)                         # [B, T, 64]

            # -- Kinematics (velocity + acceleration from keypoints) --
            if self.use_kinematics:
                kin_out = self._compute_kinematics(
                    keypoints, self.n_persons, kp_conf,
                )                                                      # [B, T, kin_dim]

            fusion_parts = [pose_out, emb_out, flow_out]
            if self.use_kinematics:
                fusion_parts.append(kin_out)
            fused = torch.cat(fusion_parts, dim=-1)

        x = self.fusion(fused)                                         # [B, T, d_model]

        # -- TCN (expects channels-first) --
        x = x.transpose(1, 2)                                         # [B, d_model, T]
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)                                         # [B, T, d_model]

        # -- Output --
        out = self.output_head(x).squeeze(-1)                          # [B, T]
        return torch.sigmoid(out)

    def _compute_kinematics(
        self,
        kp: torch.Tensor,
        n_persons: int,
        kp_conf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity and acceleration features from keypoints.

        Args:
            kp: [B, T, N, K, 3] keypoints (x, y, confidence)
            n_persons: number of persons in the N dimension
            kp_conf: [B, T, N] confidence scores
        Returns: [B, T, kin_dim]
        """
        pos_xy = kp[:, :, :, :, :2]                        # [B, T, N, K, 2]
        vel = torch.zeros_like(pos_xy)
        vel[:, 1:] = pos_xy[:, 1:] - pos_xy[:, :-1]        # first derivative
        acc = torch.zeros_like(pos_xy)
        acc[:, 2:] = vel[:, 2:] - vel[:, 1:-1]              # second derivative
        kin = torch.cat([vel, acc], dim=-1)                  # [B, T, N, K, 4]
        B, T = kp.shape[:2]
        kin = kin.reshape(B, T, n_persons, -1)               # [B, T, N, K*4]
        kin_feat = self.kin_encoder(kin)                      # [B, T, N, kin_dim]
        return self.kin_attn(kin_feat, kp_conf)               # [B, T, kin_dim]

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
