"""Lite dense-flow TCN for funscript prediction.

This variant is designed for multiclass scene tensors with 5 partner slots and
1 beholder slot. It keeps the dense RAFT flow map as a first-class input,
samples motion vectors at the selected keypoint locations, and uses a lightly
gated embedding residual so the semantic branch cannot dominate the spatial one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn import TCNBlock


MODEL_CONFIG_KEYS = {
    "d_model",
    "n_blocks",
    "dropout",
    "n_partners",
    "n_beholders",
    "n_keypoints",
    "n_beholder_keypoints",
    "embed_dim",
    "flow_mode",
    "flow_dense_size",
    "performer_hidden_dim",
    "beholder_hidden_dim",
    "flow_global_dim",
    "pair_dim",
    "flow_flip",
}

BEHOLDER_VERTICAL_SAMPLE_Y = [0.17, 0.25, 0.33, 0.42, 0.50, 0.58, 0.67, 0.75, 0.83, 0.92]


def extract_lite_model_config(config: dict[str, object]) -> dict[str, object]:
    """Filter checkpoint config down to FunscriptLiteTCN constructor kwargs."""
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


class _DenseFlowEncoder(nn.Module):
    """Encodes global dense optical flow maps into a flat feature vector."""
    def __init__(self, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _, height, width = flow.shape  # [B, T, 2, H, W]
        # Only uses the Y-component of the flow (index 1) since we primarily care about vertical motion
        x = flow[:, :, 1:2].reshape(bsz * seq_len, 1, height, width)
        x = self.net(x)
        return x.view(bsz, seq_len, -1)


class _KeypointFlowEncoder(nn.Module):
    """Encodes a person's spatial presence by combining their keypoint geometry
    and the optical flow vectors sampled at those keypoint locations.
    """
    def __init__(
        self,
        n_keypoints: int,
        point_dim: int,
        hidden_dim: int,
        dropout: float,
        n_extra_points: int = 0,
    ):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.n_extra_points = n_extra_points
        self.point_proj = nn.Sequential(
            nn.Linear(6, point_dim),
            nn.LayerNorm(point_dim),
            nn.GELU(),
        )
        self.person_proj = nn.Sequential(
            nn.Linear(((n_keypoints + n_extra_points) * point_dim) + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        sampled_flow: torch.Tensor,
        extra_xy: torch.Tensor | None = None,
        extra_sampled_flow: torch.Tensor | None = None,
        extra_conf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xy = keypoints[..., :2]  # [B, T, K, 2]
        conf = keypoints[..., 2:3]  # [B, T, K, 1]
        
        # Calculate the center of mass (root) of visible keypoints
        weights = conf.clamp_min(0.0)
        root = (xy * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)  # [B, T, 2]
        rel_xy = xy - root.unsqueeze(2)  # [B, T, K, 2]
        
        # Fuse absolute pos, relative pos, confidence, and local motion (flow) for each point
        point_inputs = torch.cat([xy, rel_xy, conf, sampled_flow], dim=-1)  # [B, T, K, 6]
        point_features = self.point_proj(point_inputs) * conf  # [B, T, K, D]

        # Process additional non-keypoint spatial samples if provided (e.g. for beholder vertical strip)
        if extra_xy is not None and extra_sampled_flow is not None and extra_conf is not None:
            extra_rel_xy = extra_xy - root.unsqueeze(2)  # [B, T, E, 2]
            extra_inputs = torch.cat([extra_xy, extra_rel_xy, extra_conf, extra_sampled_flow], dim=-1)
            extra_features = self.point_proj(extra_inputs) * extra_conf
            point_features = torch.cat([point_features, extra_features], dim=2)

        # Aggregate overall summary statistics for the person
        visible_ratio = (conf > 0).to(conf.dtype).mean(dim=2)  # [B, T, 1]
        mean_conf = conf.mean(dim=2)  # [B, T, 1]
        summary = torch.cat([root, visible_ratio, mean_conf], dim=-1)  # [B, T, 4]
        
        flat = point_features.flatten(start_dim=2)  # [B, T, K*D]
        return self.person_proj(torch.cat([flat, summary], dim=-1))


class FunscriptLiteTCN(nn.Module):
    """Dense-flow lite TCN with flow-aligned keypoint sampling.

    Architecture & Data Merging Process:
    ------------------------------------
    1. Parsing Inputs: Receives keypoints and semantic embeddings for multiple candidates
       (partners) and a camera view (beholder), along with dense optical flow maps.
    2. Primary Selection: Dynamically identifies the main active performer out of all partners
       based on maximum visibility and proximity to frame center.
    3. Motion Sampling: Extracts global motion context from the dense flow map, and simultaneously
       samples pinpoint local motion vectors from the flow map precisely at the coordinates
       of the performer's and beholder's keypoints.
    4. Independent Encoding:
       - Spatial encoders process local geometry + local flow for both the performer and beholder.
       - A Pair encoder computes relative interactions (distance, root movement) between them.
       - Global flow is compressed via CNN.
    5. Feature Fusion: All spatial/motion features are fused into a main `spatial` vector.
       Semantic embeddings (like YOLO/CLIP features) are separately compressed and then combined
       with the spatial vector using a learned gating mechanism. This prevents noisy semantic
       data from overriding reliable physical motion data.
    6. Temporal Processing: The fused sequence passes through a Bidirectional GRU to capture
       overall context and smooth out anomalies, followed by a stack of Temporal Convolutional
       Networks (TCN) with exponentially increasing dilation to model complex, fast motion
       dynamics across a massive temporal receptive field.
    7. Output Projection: An MLP head projects the main hidden dimension down to a final sequence
       of 1D positions, normalized to [0, 1] bounds using a sigmoid.

    Model Arguments:
    ----------------
    d_model: Dimensionality of the main internal hidden state (TCN, GRU). Increasing it improves
             representational power but slows down training/inference and increases risk of overfitting.
    n_blocks: Number of dilated TCN blocks. More blocks exponentially increase the temporal receptive
              field (how far back/forward the model looks) but increase parameter count.
    dropout: Regularization probability used throughout the network to prevent overfitting.
    n_partners: The maximum number of performer candidates in the input tensor. Allows the model to
                handle scenes with multiple people and choose the primary one.
    n_beholders: The number of camera perspectives / POV slots. Must be >= 1.
    n_keypoints: The number of anatomical points tracked for performers (e.g., 21 for full body).
                 Changing this requires retraining and matching the upstream pose tracker.
    n_beholder_keypoints: Number of points tracked for the beholder perspective.
    embed_dim: The dimensionality of the input semantic embeddings (e.g. YOLO/ViT features).
    flow_mode: The flow representation type. Must be "dense".
    flow_dense_size: Expected spatial resolution of the dense flow map (e.g. 32x32). Lower resolution
                     is faster but loses fine-grained global context.
    performer_hidden_dim: Dim of the output feature vector representing the primary performer's motion/pose.
    beholder_hidden_dim: Dim of the output feature vector representing the camera/beholder's motion/pose.
    flow_global_dim: Dim of the global optical flow feature extracted by the CNN.
    pair_dim: Dim of the feature vector capturing relative interaction between performer and beholder.
    flow_flip: Augmentation flag ("none", "hflip", "vflip", "hvflip") dictating how flow is mirrored.
               Useful for test-time augmentation or training data multiplication.
    """

    VALID_FLOW_FLIPS = {"none", "hflip", "vflip", "hvflip"}

    def __init__(
        self,
        d_model: int = 192,
        n_blocks: int = 4,
        dropout: float = 0.15,
        n_partners: int = 5,
        n_beholders: int = 1,
        n_keypoints: int = 21,
        n_beholder_keypoints: int = 7,
        embed_dim: int = 512,
        flow_mode: str = "dense",
        flow_dense_size: int = 32,
        performer_hidden_dim: int = 96,
        beholder_hidden_dim: int = 64,
        flow_global_dim: int = 48,
        pair_dim: int = 32,
        flow_flip: str = "none",
    ):
        super().__init__()
        if flow_mode != "dense":
            raise ValueError("FunscriptLiteTCN requires flow_mode='dense'")
        if flow_flip not in self.VALID_FLOW_FLIPS:
            raise ValueError(f"Invalid flow_flip: {flow_flip}")
        if n_beholders < 1:
            raise ValueError("FunscriptLiteTCN requires at least one beholder slot")

        self.d_model = d_model
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.n_partners = n_partners
        self.n_beholders = n_beholders
        self.n_keypoints = n_keypoints
        self.n_beholder_keypoints = n_beholder_keypoints
        self.embed_dim = embed_dim
        self.flow_mode = flow_mode
        self.flow_dense_size = flow_dense_size
        self.performer_hidden_dim = performer_hidden_dim
        self.beholder_hidden_dim = beholder_hidden_dim
        self.flow_global_dim = flow_global_dim
        self.pair_dim = pair_dim
        self.flow_flip = flow_flip
        self.register_buffer(
            "beholder_vertical_sample_y",
            torch.tensor(BEHOLDER_VERTICAL_SAMPLE_Y, dtype=torch.float32),
            persistent=False,
        )

        self.performer_encoder = _KeypointFlowEncoder(
            n_keypoints=n_keypoints,
            point_dim=24,
            hidden_dim=performer_hidden_dim,
            dropout=dropout,
        )
        self.beholder_encoder = _KeypointFlowEncoder(
            n_keypoints=n_beholder_keypoints,
            point_dim=16,
            hidden_dim=beholder_hidden_dim,
            dropout=dropout,
            n_extra_points=len(BEHOLDER_VERTICAL_SAMPLE_Y),
        )
        self.flow_encoder = _DenseFlowEncoder(flow_global_dim, dropout)
        self.pair_encoder = nn.Sequential(
            nn.Linear(7, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        spatial_in = performer_hidden_dim + beholder_hidden_dim + flow_global_dim + pair_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.performer_emb_encoder = nn.Sequential(
            nn.Linear(embed_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.beholder_emb_encoder = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Gating mechanism ensures semantics only influence the output if they agree with spatial cues
        self.embedding_gate = nn.Linear(d_model, d_model)
        self.embedding_scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.fuse_norm = nn.LayerNorm(d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.post_gru_norm = nn.LayerNorm(d_model)
        
        # Stack of dilated temporal convolutions to build massive receptive field
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size=3, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def set_flow_flip(self, flow_flip: str) -> None:
        """Update the flow sampling flip mode at runtime."""
        if flow_flip not in self.VALID_FLOW_FLIPS:
            raise ValueError(f"Invalid flow_flip: {flow_flip}")
        self.flow_flip = flow_flip

    def forward(
        self,
        keypoints: torch.Tensor,
        embeddings: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = keypoints.shape[:2]

        flow = self._apply_flow_flip(flow)  # [B, T, 2, H, W]
        partner_kp = keypoints[:, :, :self.n_partners]  # [B, T, Np, K, 3]
        partner_emb = embeddings[:, :, :self.n_partners]  # [B, T, Np, E]

        # 1. Select the primary performer based on visibility/center distance
        primary_idx = self._select_primary_partner(partner_kp)  # [B, T]
        performer_kp = self._gather_partner_keypoints(partner_kp, primary_idx)  # [B, T, K, 3]
        performer_emb = self._gather_partner_embeddings(partner_emb, primary_idx)  # [B, T, E]

        # 2. Extract beholder data
        beholder_slot = self.n_partners
        beholder_kp = keypoints[:, :, beholder_slot, :self.n_beholder_keypoints]  # [B, T, Kb, 3]
        beholder_emb = embeddings[:, :, beholder_slot]  # [B, T, E]

        # 3. Sample dense flow at specific locations
        beholder_extra_xy, beholder_extra_conf = self._build_beholder_vertical_samples(beholder_kp)
        performer_flow = self._sample_flow_vectors(flow, performer_kp[..., :2])  # [B, T, K, 1]
        beholder_flow = self._sample_flow_vectors(flow, beholder_kp[..., :2])  # [B, T, Kb, 1]
        beholder_extra_flow = self._sample_flow_vectors(flow, beholder_extra_xy)  # [B, T, 10, 1]
        
        # 4. Feature Extraction & Encoding
        flow_global = self.flow_encoder(flow)  # [B, T, D]
        performer_feat = self.performer_encoder(performer_kp, performer_flow)  # [B, T, Dp]
        beholder_feat = self.beholder_encoder(
            beholder_kp,
            beholder_flow,
            extra_xy=beholder_extra_xy,
            extra_sampled_flow=beholder_extra_flow,
            extra_conf=beholder_extra_conf,
        )  # [B, T, Db]
        pair_feat = self._encode_pair_features(flow, performer_kp, beholder_kp)  # [B, T, Dpair]

        # 5. Spatial Fusion
        spatial = self.spatial_proj(
            torch.cat([performer_feat, beholder_feat, flow_global, pair_feat], dim=-1),
        )  # [B, T, d_model]

        # 6. Semantic Gating & Addition
        emb_latent = self.embedding_proj(torch.cat([
            self.performer_emb_encoder(performer_emb),
            self.beholder_emb_encoder(beholder_emb),
        ], dim=-1))  # [B, T, d_model]
        emb_gate = torch.sigmoid(self.embedding_gate(spatial))
        emb_scale = torch.sigmoid(self.embedding_scale_logit)

        x = self.fuse_norm(spatial + (emb_scale * emb_gate * emb_latent))  # [B, T, d_model]
        
        # 7. Temporal Processing
        gru_out, _ = self.gru(x)
        x = self.post_gru_norm(x + gru_out)  # [B, T, d_model]

        x = x.transpose(1, 2)  # [B, d_model, T]
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # [B, T, d_model]

        # 8. Final Position Output
        out = self.output_head(x).squeeze(-1)  # [B, T]
        return torch.sigmoid(out).unsqueeze(1)  # [B, 1, T]

    def count_parameters(self) -> dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {"total": total, "trainable": trainable}

    def _apply_flow_flip(self, flow: torch.Tensor) -> torch.Tensor:
        if self.flow_flip == "none":
            return flow

        out = flow
        if self.flow_flip in {"hflip", "hvflip"}:
            out = torch.flip(out, dims=(-1,))
            out = out.clone()
            out[:, :, 0] = -out[:, :, 0]
        if self.flow_flip in {"vflip", "hvflip"}:
            out = torch.flip(out, dims=(-2,))
            out = out.clone()
            out[:, :, 1] = -out[:, :, 1]
        return out

    def _select_primary_partner(self, partner_kp: torch.Tensor) -> torch.Tensor:
        conf = partner_kp[..., 2]  # [B, T, Np, K]
        visible = (conf > 0).sum(dim=-1)  # [B, T, Np]
        weights = conf.clamp_min(0.0).unsqueeze(-1)
        centers = (partner_kp[..., :2] * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1e-6)
        dist = torch.linalg.vector_norm(centers - 0.5, dim=-1)  # [B, T, Np]

        max_visible = visible.max(dim=-1, keepdim=True).values
        tie_scores = torch.where(
            visible == max_visible,
            -dist,
            torch.full_like(dist, -1e6),
        )
        return tie_scores.argmax(dim=-1)

    def _gather_partner_keypoints(self, partner_kp: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        gather_index = index.view(index.shape[0], index.shape[1], 1, 1, 1)
        gather_index = gather_index.expand(-1, -1, 1, self.n_keypoints, 3)
        return partner_kp.gather(2, gather_index).squeeze(2)

    def _gather_partner_embeddings(self, partner_emb: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        gather_index = index.view(index.shape[0], index.shape[1], 1, 1)
        gather_index = gather_index.expand(-1, -1, 1, self.embed_dim)
        return partner_emb.gather(2, gather_index).squeeze(2)

    def _sample_flow_vectors(self, flow: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _, height, width = flow.shape
        n_points = xy.shape[2]

        coords = xy.clamp(0.0, 1.0)
        grid = (coords * 2.0) - 1.0  # [B, T, P, 2]
        flow_bt = flow[:, :, 1:2].reshape(bsz * seq_len, 1, height, width)
        grid_bt = grid.reshape(bsz * seq_len, n_points, 1, 2)
        sampled = F.grid_sample(
            flow_bt,
            grid_bt,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)  # [B*T, P, 1]
        return sampled.view(bsz, seq_len, n_points, 1)

    def _encode_pair_features(
        self,
        flow: torch.Tensor,
        performer_kp: torch.Tensor,
        beholder_kp: torch.Tensor,
    ) -> torch.Tensor:
        performer_root, performer_conf = self._person_root_and_conf(performer_kp)
        beholder_root, beholder_conf = self._person_root_and_conf(beholder_kp)

        performer_root_flow = self._sample_flow_vectors(flow, performer_root.unsqueeze(2)).squeeze(2)
        beholder_root_flow = self._sample_flow_vectors(flow, beholder_root.unsqueeze(2)).squeeze(2)
        performer_visible = (performer_kp[..., 2] > 0).to(performer_kp.dtype).mean(dim=-1, keepdim=True)
        beholder_visible = (beholder_kp[..., 2] > 0).to(beholder_kp.dtype).mean(dim=-1, keepdim=True)

        pair_inputs = torch.cat([
            performer_root - beholder_root,
            performer_root_flow - beholder_root_flow,
            performer_conf,
            beholder_conf,
            performer_visible,
            beholder_visible,
        ], dim=-1)  # [B, T, 7]
        return self.pair_encoder(pair_inputs)

    def _build_beholder_vertical_samples(
        self,
        beholder_kp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pelvis_idx = 4
        root_xy, root_conf = self._person_root_and_conf(beholder_kp)
        fallback_x = torch.where(
            root_conf > 1e-6,
            root_xy[..., :1],
            torch.full_like(root_xy[..., :1], 0.5),
        )

        beholder_conf = beholder_kp[..., 2:3].amax(dim=2)  # [B, T, 1]
        if pelvis_idx < beholder_kp.shape[2]:
            pelvis_x = beholder_kp[..., pelvis_idx, :1]
            pelvis_conf = beholder_kp[..., pelvis_idx, 2:3]
            sample_x = torch.where(pelvis_conf > 1e-6, pelvis_x, fallback_x)
            sample_conf = torch.where(pelvis_conf > 1e-6, pelvis_conf, beholder_conf)
        else:
            sample_x = fallback_x
            sample_conf = beholder_conf

        sample_y = self.beholder_vertical_sample_y.view(1, 1, -1, 1).to(device=beholder_kp.device, dtype=beholder_kp.dtype)
        sample_x = sample_x.unsqueeze(2).expand(-1, -1, sample_y.shape[2], -1)
        sample_conf = sample_conf.unsqueeze(2).expand(-1, -1, sample_y.shape[2], -1)
        sample_xy = torch.cat([sample_x, sample_y.expand_as(sample_x)], dim=-1)
        return sample_xy, sample_conf

    def _person_root_and_conf(self, keypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xy = keypoints[..., :2]
        conf = keypoints[..., 2:3]
        weights = conf.clamp_min(0.0)
        root = (xy * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)
        mean_conf = conf.mean(dim=2)
        return root, mean_conf
