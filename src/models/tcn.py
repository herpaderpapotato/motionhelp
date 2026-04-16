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


COCO_17_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16),
]

PERFORMER_21_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 17), (12, 17), (17, 18),
    (5, 19), (6, 20), (19, 20), (18, 19), (18, 20),
]

BEHOLDER_7_BONES = [
    (0, 1),
    (0, 2), (1, 3),
    (0, 4), (1, 4),
    (4, 5),
]

MODEL_CONFIG_KEYS = {
    "d_model",
    "n_blocks",
    "kernel_size",
    "dropout",
    "n_persons",
    "n_keypoints",
    "embed_dim",
    "flow_dim",
    "flow_mode",
    "flow_dense_size",
    "n_partners",
    "n_beholders",
    "n_beholder_keypoints",
    "beholder_pose_dim",
    "beholder_emb_dim",
    "use_kinematics",
    "use_ddl",
    "kin_dim",
    "use_gated_fusion",
    "use_difference_pathway",
    "difference_dim",
}


def extract_model_config(config: dict[str, object]) -> dict[str, object]:
    """Filter checkpoint config down to FunscriptTCN constructor kwargs."""
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


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


class StructuredPoseEncoder(nn.Module):
    """Encode per-person pose using relative joints, bones, and velocities."""

    def __init__(
        self,
        n_keypoints: int,
        output_dim: int,
        kernel_size: int,
        dropout: float,
        bones: list[tuple[int, int]],
        root_index: int | None,
        root_pair: tuple[int, int] | None,
    ):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.root_index = root_index
        self.root_pair = root_pair

        if bones:
            bone_tensor = torch.tensor(bones, dtype=torch.long)
            self.register_buffer("bone_start", bone_tensor[:, 0], persistent=False)
            self.register_buffer("bone_end", bone_tensor[:, 1], persistent=False)
        else:
            self.register_buffer("bone_start", torch.zeros(0, dtype=torch.long), persistent=False)
            self.register_buffer("bone_end", torch.zeros(0, dtype=torch.long), persistent=False)

        feature_dim = (n_keypoints * 2) + (len(bones) * 2) + (n_keypoints * 2) + n_keypoints + 2
        self.frame_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        pad1 = (kernel_size - 1) // 2
        pad2 = 2 * (kernel_size - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size, padding=pad1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size, padding=pad2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, kp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pose features before person pooling.

        Args:
            kp: [B, T, N, K, 3]
        Returns:
            encoded: [B, T, N, D]
            person_conf: [B, T, N]
        """
        xy = kp[..., :2]                                                 # [B, T, N, K, 2]
        conf = kp[..., 2]                                                # [B, T, N, K]

        root_xy = self._compute_root(xy, conf)                           # [B, T, N, 2]
        rel_xy = (xy - root_xy.unsqueeze(-2)) * conf.unsqueeze(-1)       # [B, T, N, K, 2]

        rel_vel = torch.zeros_like(rel_xy)
        if rel_xy.shape[1] > 1:
            rel_vel[:, 1:] = rel_xy[:, 1:] - rel_xy[:, :-1]

        root_vel = torch.zeros_like(root_xy)
        if root_xy.shape[1] > 1:
            root_vel[:, 1:] = root_xy[:, 1:] - root_xy[:, :-1]

        bone_vec = self._compute_bones(rel_xy)                           # [B, T, N, B, 2]
        person_conf = self._mean_visible_conf(conf)                      # [B, T, N]

        features = torch.cat(
            [
                rel_xy.flatten(start_dim=3),
                bone_vec.flatten(start_dim=3),
                rel_vel.flatten(start_dim=3),
                conf,
                root_vel,
            ],
            dim=-1,
        )                                                                # [B, T, N, F]
        encoded = self.frame_proj(features)                              # [B, T, N, D]

        bsz, seq_len, n_persons, feat_dim = encoded.shape
        encoded_bt = encoded.permute(0, 2, 3, 1).reshape(bsz * n_persons, feat_dim, seq_len)
        encoded_bt = encoded_bt + self.temporal(encoded_bt)
        encoded = encoded_bt.reshape(bsz, n_persons, feat_dim, seq_len).permute(0, 3, 1, 2)
        encoded = self.out_norm(encoded)
        return encoded, person_conf

    def _compute_root(self, xy: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Infer a stable per-person root point using pelvis or hip midpoint."""
        root_xy = None
        root_conf = None

        if self.root_index is not None and self.root_index < self.n_keypoints:
            root_xy = xy[..., self.root_index, :]
            root_conf = conf[..., self.root_index]

        if self.root_pair is not None and max(self.root_pair) < self.n_keypoints:
            pair_xy = 0.5 * (xy[..., self.root_pair[0], :] + xy[..., self.root_pair[1], :])
            pair_conf = 0.5 * (conf[..., self.root_pair[0]] + conf[..., self.root_pair[1]])
            if root_xy is None:
                root_xy = pair_xy
                root_conf = pair_conf
            else:
                use_pair = root_conf <= 1e-6
                root_xy = torch.where(use_pair.unsqueeze(-1), pair_xy, root_xy)
                root_conf = torch.where(use_pair, pair_conf, root_conf)

        if root_xy is None:
            weights = conf.clamp_min(1e-6).unsqueeze(-1)
            return (xy * weights).sum(dim=3) / weights.sum(dim=3).clamp_min(1e-6)

        if root_conf is not None:
            missing_root = root_conf <= 1e-6
            if missing_root.any():
                weights = conf.clamp_min(1e-6).unsqueeze(-1)
                fallback = (xy * weights).sum(dim=3) / weights.sum(dim=3).clamp_min(1e-6)
                root_xy = torch.where(missing_root.unsqueeze(-1), fallback, root_xy)
        return root_xy

    def _compute_bones(self, rel_xy: torch.Tensor) -> torch.Tensor:
        if self.bone_start.numel() == 0:
            shape = rel_xy.shape[:3] + (0, 2)
            return rel_xy.new_zeros(shape)
        bone_start = rel_xy.index_select(3, self.bone_start)
        bone_end = rel_xy.index_select(3, self.bone_end)
        return bone_end - bone_start

    @staticmethod
    def _mean_visible_conf(conf: torch.Tensor) -> torch.Tensor:
        visible = conf > 0
        denom = visible.sum(dim=-1).clamp_min(1)
        return conf.sum(dim=-1) / denom


class DifferenceMagnitudeEncoder(nn.Module):
    """Encode beholder-to-performer keypoint difference magnitudes."""

    def __init__(
        self,
        n_beholder_keypoints: int,
        n_partners: int,
        n_keypoints: int,
        output_dim: int,
        kernel_size: int,
        dropout: float,
        pair_dim: int = 8,
    ):
        super().__init__()
        self.n_beholder_keypoints = n_beholder_keypoints
        self.n_partners = n_partners
        self.n_keypoints = n_keypoints
        self.output_dim = output_dim
        self.pair_dim = pair_dim

        feature_count = n_beholder_keypoints * n_partners * n_keypoints
        self.pair_proj = nn.Sequential(
            nn.Linear(2, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
        )
        self.frame_proj = nn.Sequential(
            nn.Linear(feature_count * pair_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        pad1 = (kernel_size - 1) // 2
        pad2 = 2 * (kernel_size - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size, padding=pad1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size, padding=pad2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, partner_kp: torch.Tensor, beholder_kp: torch.Tensor) -> torch.Tensor:
        """Encode per-frame difference features.

        Args:
            partner_kp: [B, T, Np, Kp, 3]
            beholder_kp: [B, T, Nb, Kb, 3]
        Returns: [B, T, D]
        """
        bsz, seq_len = partner_kp.shape[:2]
        if beholder_kp.shape[2] == 0:
            return partner_kp.new_zeros((bsz, seq_len, self.output_dim))

        partner_xy = partner_kp[..., :2]                              # [B, T, Np, Kp, 2]
        partner_conf = partner_kp[..., 2]                             # [B, T, Np, Kp]
        beholder_xy = beholder_kp[..., :2]                            # [B, T, Nb, Kb, 2]
        beholder_conf = beholder_kp[..., 2]                           # [B, T, Nb, Kb]

        collapsed_beh_xy, collapsed_beh_conf = self._collapse_beholders(
            beholder_xy,
            beholder_conf,
        )                                                             # [B, T, Kb, 2], [B, T, Kb]

        diff = (
            partner_xy.unsqueeze(2)
            - collapsed_beh_xy.unsqueeze(3).unsqueeze(4)
        )                                                             # [B, T, Kb, Np, Kp, 2]
        conf_prod = (
            collapsed_beh_conf.unsqueeze(3).unsqueeze(4).unsqueeze(-1)
            * partner_conf.unsqueeze(2).unsqueeze(-1)
        )                                                             # [B, T, Kb, Np, Kp, 1]
        valid_mask = (conf_prod > 0).to(diff.dtype)
        diff_mag = torch.linalg.vector_norm(diff, dim=-1, keepdim=True) * valid_mask

        pair_features = torch.cat([diff_mag, conf_prod], dim=-1)      # [B, T, Kb, Np, Kp, 2]
        pair_encoded = self.pair_proj(pair_features)                   # [B, T, Kb, Np, Kp, pair_dim]
        frame_features = pair_encoded.reshape(bsz, seq_len, -1)       # [B, T, Kb*Np*Kp*pair_dim]
        encoded = self.frame_proj(frame_features)                      # [B, T, D]

        encoded_bt = encoded.transpose(1, 2)                           # [B, D, T]
        encoded_bt = encoded_bt + self.temporal(encoded_bt)
        return self.out_norm(encoded_bt.transpose(1, 2))               # [B, T, D]

    @staticmethod
    def _collapse_beholders(
        beholder_xy: torch.Tensor,
        beholder_conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse multiple beholders down to one per frame/keypoint."""
        if beholder_xy.shape[2] == 1:
            return beholder_xy.squeeze(2), beholder_conf.squeeze(2)

        weights = beholder_conf.clamp_min(0.0)
        denom = weights.sum(dim=2, keepdim=False).clamp_min(1e-6)
        xy = (beholder_xy * weights.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)
        conf = weights.max(dim=2).values
        return xy, conf


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


class GatedFusion(nn.Module):
    """Context-aware gated multimodal fusion.

    Each modality stream gets a learned sigmoid gate that sees the full
    cross-modal context, allowing per-frame dynamic suppression of noisy
    or irrelevant modalities. Inspired by:
      - Gated Fusion Networks (Ahmad et al. 2025): per-modality gating
      - Gated Recursive Fusion (Shihata 2025): cross-modal information flow

    Replaces the concat→Linear baseline fusion with:
        context = cat(streams)          # full cross-modal view
        g_m = sigmoid(W_m · context)    # per-modality gate
        gated_m = g_m ⊙ stream_m       # gated features
        output = proj(cat(gated_m))     # project to d_model
    """

    def __init__(self, stream_dims: list[int], d_model: int):
        super().__init__()
        total_in = sum(stream_dims)
        self.gates = nn.ModuleList([
            nn.Linear(total_in, dim) for dim in stream_dims
        ])
        self.proj = nn.Sequential(
            nn.Linear(total_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            streams: list of [B, T, dim_i] tensors, one per modality
        Returns: [B, T, d_model] fused representation
        """
        context = torch.cat(streams, dim=-1)             # [B, T, total_in]
        gated = []
        for stream, gate_fn in zip(streams, self.gates):
            gate = torch.sigmoid(gate_fn(context))       # [B, T, dim_i]
            gated.append(gate * stream)                  # [B, T, dim_i]
        return self.proj(torch.cat(gated, dim=-1))       # [B, T, d_model]


class FlowSpatialEncoder(nn.Module):
    """Per-frame spatial encoder for dense optical flow maps.

    Takes [B, T, 2, H, W] dense flow and produces [B, T, out_dim] tokens
    using a small 2D CNN that preserves spatial structure.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: [B, T, 2, H, W] dense flow maps.
        Returns: [B, T, out_dim]
        """
        b, t, c, h, w = flow.shape
        x = flow.reshape(b * t, c, h, w)          # [B*T, 2, H, W]
        x = self.net(x)                            # [B*T, 64, 1, 1]
        x = self.proj(x)                           # [B*T, out_dim]
        return x.view(b, t, -1)                    # [B, T, out_dim]


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

    Output: [B, 4, T] position values in [0, 1]
        Channel 0: fused (all modalities through main TCN backbone)
        Channel 1: pose-only auxiliary branch
        Channel 2: embedding-only auxiliary branch
        Channel 3: flow-only auxiliary branch
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
        flow_mode: str = "summary",
        flow_dense_size: int = 32,
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
        use_gated_fusion: bool = False,
        use_difference_pathway: bool = False,
        difference_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_keypoints = n_keypoints
        self.embed_dim = embed_dim
        self.flow_dim = flow_dim
        self.flow_mode = flow_mode
        self.flow_dense_size = flow_dense_size
        self.multiclass = n_partners is not None
        self.use_kinematics = use_kinematics
        self.use_ddl = use_ddl
        self.kin_dim = kin_dim
        self.use_gated_fusion = use_gated_fusion
        self.use_difference_pathway = use_difference_pathway and self.multiclass
        self.difference_dim = difference_dim
        self._kernel_size = kernel_size
        self._dropout = dropout

        if self.multiclass:
            self.n_partners = n_partners
            self.n_beholders = n_beholders
            self.n_persons = n_partners  # PersonAttention uses partner count
            self.n_beholder_keypoints = n_beholder_keypoints
            self.beholder_pose_dim = beholder_pose_dim
            self.beholder_emb_dim = beholder_emb_dim
        else:
            self.n_persons = n_persons

        pose_bones, pose_root_index, pose_root_pair = self._performer_topology(n_keypoints)

        # -- Per-person feature encoders (partners in multiclass, all in single) --
        self.pose_encoder = StructuredPoseEncoder(
            n_keypoints=n_keypoints,
            output_dim=128,
            kernel_size=kernel_size,
            dropout=dropout,
            bones=pose_bones,
            root_index=pose_root_index,
            root_pair=pose_root_pair,
        )
        self.pose_attn = PersonAttention(128)

        self.emb_encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.emb_attn = PersonAttention(128)

        flow_out_dim = 64
        if self.flow_mode == "dense":
            self.flow_encoder = FlowSpatialEncoder(out_dim=flow_out_dim)
        else:
            self.flow_encoder = nn.Sequential(
                nn.Linear(flow_dim, flow_out_dim),
                nn.LayerNorm(flow_out_dim),
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
            beh_bones, beh_root_index, beh_root_pair = self._beholder_topology(n_beholder_keypoints)
            self.beholder_pose_encoder = StructuredPoseEncoder(
                n_keypoints=n_beholder_keypoints,
                output_dim=beholder_pose_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                bones=beh_bones,
                root_index=beh_root_index,
                root_pair=beh_root_pair,
            )
            self.beholder_emb_encoder = nn.Sequential(
                nn.Linear(embed_dim, beholder_emb_dim),
                nn.LayerNorm(beholder_emb_dim),
                nn.GELU(),
            )
            stream_dims = [128, 128, 64, beholder_pose_dim, beholder_emb_dim]
            if self.use_difference_pathway:
                self.difference_encoder = DifferenceMagnitudeEncoder(
                    n_beholder_keypoints=n_beholder_keypoints,
                    n_partners=n_partners,
                    n_keypoints=n_keypoints,
                    output_dim=difference_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                stream_dims.append(difference_dim)
            if self.use_kinematics:
                stream_dims.append(kin_dim)
            fusion_in = sum(stream_dims)
        else:
            stream_dims = [128, 128, 64]
            if self.use_kinematics:
                stream_dims.append(kin_dim)
            fusion_in = sum(stream_dims)

        # -- Feature fusion --
        if self.use_gated_fusion:
            self.fusion = GatedFusion(stream_dims, d_model)
        else:
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

        # -- Output head (main fused path) --
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # -- Auxiliary per-modality branches --
        # Each branch: project → small TCN (3 blocks) → output head
        aux_d = d_model // 4  # 64 by default
        aux_n_blocks = min(3, n_blocks)
        aux_dilations = [2 ** i for i in range(aux_n_blocks)]  # [1, 2, 4]

        # Pose auxiliary branch (input: 128-d from pose attention pooling)
        self.aux_pose_proj = nn.Sequential(
            nn.Linear(128, aux_d),
            nn.LayerNorm(aux_d),
            nn.GELU(),
        )
        self.aux_pose_tcn = nn.ModuleList([
            block_cls(aux_d, kernel_size, d, dropout) for d in aux_dilations
        ])
        self.aux_pose_head = nn.Sequential(
            nn.Linear(aux_d, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Embedding auxiliary branch (input: 128-d from emb attention pooling)
        self.aux_emb_proj = nn.Sequential(
            nn.Linear(128, aux_d),
            nn.LayerNorm(aux_d),
            nn.GELU(),
        )
        self.aux_emb_tcn = nn.ModuleList([
            block_cls(aux_d, kernel_size, d, dropout) for d in aux_dilations
        ])
        self.aux_emb_head = nn.Sequential(
            nn.Linear(aux_d, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Flow auxiliary branch (input: 64-d from flow encoder)
        self.aux_flow_proj = nn.Sequential(
            nn.Linear(flow_out_dim, aux_d),
            nn.LayerNorm(aux_d),
            nn.GELU(),
        )
        self.aux_flow_tcn = nn.ModuleList([
            block_cls(aux_d, kernel_size, d, dropout) for d in aux_dilations
        ])
        self.aux_flow_head = nn.Sequential(
            nn.Linear(aux_d, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
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
            flow:       [B, T, F] (summary mode) or [B, T, 2, H, W] (dense mode)
        Returns: [B, 4, T] positions in [0, 1]
            Channel 0: fused, 1: pose-only, 2: emb-only, 3: flow-only
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
            pose_feat, kp_conf = self.pose_encoder(partner_kp)         # [B, T, Np, 128], [B, T, Np]
            pose_out = self.pose_attn(pose_feat, kp_conf)              # [B, T, 128]

            # -- Partner embeddings (existing path) --
            emb_feat = self.emb_encoder(partner_emb)                   # [B, T, Np, 128]
            emb_out = self.emb_attn(emb_feat, kp_conf)                # [B, T, 128]

            # -- Beholder pose --
            beh_pose_feat, beh_conf = self.beholder_pose_encoder(beholder_kp)  # [B, T, Nb, Db]
            beh_pose_out = beh_pose_feat.mean(dim=2)                   # [B, T, beh_pose_dim]

            # -- Beholder embeddings --
            beh_emb_feat = self.beholder_emb_encoder(beholder_emb)     # [B, T, Nb, beh_emb_dim]
            beh_emb_out = beh_emb_feat.mean(dim=2)                     # [B, T, beh_emb_dim]

            # -- Beholder/performer keypoint difference pathway --
            if self.use_difference_pathway:
                diff_out = self.difference_encoder(partner_kp, beholder_kp)  # [B, T, difference_dim]

            # -- Flow --
            flow_out = self.flow_encoder(flow)                         # [B, T, 64]

            # -- Kinematics (velocity + acceleration from partner keypoints) --
            if self.use_kinematics:
                kin_out = self._compute_kinematics(
                    partner_kp, self.n_partners, kp_conf,
                )                                                      # [B, T, kin_dim]

            # -- Fusion (augmented with beholder + optional kinematics) --
            fusion_parts = [pose_out, emb_out, flow_out, beh_pose_out, beh_emb_out]
            if self.use_difference_pathway:
                fusion_parts.append(diff_out)
            if self.use_kinematics:
                fusion_parts.append(kin_out)
        else:
            # -- Original single-class path --
            pose_feat, kp_conf = self.pose_encoder(keypoints)          # [B, T, N, 128], [B, T, N]
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

        if self.use_gated_fusion:
            x = self.fusion(fusion_parts)                              # [B, T, d_model]
        else:
            x = self.fusion(torch.cat(fusion_parts, dim=-1))           # [B, T, d_model]

        # -- Auxiliary per-modality branches (before main TCN) --
        # Pose auxiliary
        aux_p = self.aux_pose_proj(pose_out).transpose(1, 2)          # [B, aux_d, T]
        for block in self.aux_pose_tcn:
            aux_p = block(aux_p)
        aux_p = self.aux_pose_head(aux_p.transpose(1, 2)).squeeze(-1) # [B, T]

        # Embedding auxiliary
        aux_e = self.aux_emb_proj(emb_out).transpose(1, 2)            # [B, aux_d, T]
        for block in self.aux_emb_tcn:
            aux_e = block(aux_e)
        aux_e = self.aux_emb_head(aux_e.transpose(1, 2)).squeeze(-1)  # [B, T]

        # Flow auxiliary
        aux_f = self.aux_flow_proj(flow_out).transpose(1, 2)          # [B, aux_d, T]
        for block in self.aux_flow_tcn:
            aux_f = block(aux_f)
        aux_f = self.aux_flow_head(aux_f.transpose(1, 2)).squeeze(-1) # [B, T]

        # -- Main TCN backbone (expects channels-first) --
        x = x.transpose(1, 2)                                         # [B, d_model, T]
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)                                         # [B, T, d_model]

        # -- Main output --
        main_out = self.output_head(x).squeeze(-1)                     # [B, T]

        # -- Stack all channels: [fused, pose, emb, flow] --
        out = torch.stack([main_out, aux_p, aux_e, aux_f], dim=1)      # [B, 4, T]
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

    @staticmethod
    def _performer_topology(n_keypoints: int) -> tuple[list[tuple[int, int]], int | None, tuple[int, int] | None]:
        if n_keypoints >= 21:
            return PERFORMER_21_BONES, 17, (11, 12)
        if n_keypoints >= 17:
            return COCO_17_BONES, None, (11, 12)
        return [], None, None

    @staticmethod
    def _beholder_topology(n_keypoints: int) -> tuple[list[tuple[int, int]], int | None, tuple[int, int] | None]:
        if n_keypoints >= 7:
            return BEHOLDER_7_BONES, 4, (0, 1)
        if n_keypoints >= 2:
            return [], None, (0, 1)
        return [], None, None

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

