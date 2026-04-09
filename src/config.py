from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import yaml
import torch


@dataclass
class VideoConfig:
    vr_mode: bool = True
    target_fps: int = 30
    frame_size: int = 640  # resize frames to this for pose extraction
    sbs_crop: str = "left"  # "left" or "right" eye for SBS VR


# COCO 17 keypoint names (default)
_COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_COCO_KEYPOINT_BONES = [
    [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
    [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12],
    [11, 13], [12, 14], [13, 15], [14, 16],
]


@dataclass
class PoseConfig:
    model_name: str = "vrlens-finetunes-multiclass-v2-yolo11m-pose"
    model_path: str = ""  # auto-download if empty. For VR use herpaderpapotato/pose-vrlens-finetunes-multiclass
    keypoint_format: str = "coco"  # "coco" or "custom"
    keypoint_names: List[str] = field(default_factory=lambda: list(_COCO_KEYPOINT_NAMES))
    keypoint_indices: List[int] = field(default_factory=lambda: list(range(17)))
    keypoint_bones: List[List[int]] = field(default_factory=lambda: [list(b) for b in _COCO_KEYPOINT_BONES])
    confidence_threshold: float = 0.3
    batch_size: int = 32
    device: str = "auto"
    n_keypoints: int = 17


@dataclass
class FlowConfig:
    method: str = "raft"  # "raft" (GPU, recommended) or "farneback" (CPU, legacy)
    output_features: int = 64
    scale: float = 0.5  # resize factor for flow computation


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    sequence_length: int = 120  # frames per prediction window
    # Feature sources
    use_embeddings: bool = True   # YOLO model.embed features
    use_pose: bool = True        # COCO keypoints (legacy)
    use_flow: bool = True         # RAFT optical flow
    embedding_features: int = 1024  # max_persons * embed_dim
    embed_dim: int = 512            # per-person embedding dimension
    pose_features: int = 126       # max_persons × n_keypoints × 3 (x, y, conf); 2×17×3=102 for COCO, 2×21×3=126 for custom
    flow_features: int = 64
    use_scene: bool = False
    max_persons: int = 2
    output_mode: str = "direct"     # "direct" (per-frame) or "latent" (sequence-level, needs decoder)
    latent_dim: int = 32            # only used when output_mode is "latent"

@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine_restarts"  # "cosine_restarts", "cosine", or "plateau"
    warmup_epochs: int = 3
    gradient_clip: float = 1.0
    pos_loss_weight: float = 0.001
    temporal_consistency_weight: float = 0.1
    cumulative_magnitude_weight: float = 0.0
    direction_accuracy_weight: float = 0.0
    cumulative_total_weight: float = 0.0
    penalize_mid_weights: float = 0.0   # penalize predictions near 0.5 to encourage more decisive outputs
    asymmetric_position_weight: float = 0.0  # penalize predictions on wrong side of 0.5 relative to target
    variance_weight: float = 0.1
    position_loss: str = "mse"
    checkpoint_dir: str = "data/models/checkpoints"
    log_dir: str = "runs"
    save_every_n_epochs: int = 10
    val_every_n_epochs: int = 1
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class InferenceConfig:
    overlap: float = 0.5  # sliding window overlap ratio
    smoothing: str = "savgol"  # "none", "savgol", "gaussian"
    smoothing_window: int = 15
    output_rate: str = "every_frame"  # "every_frame", "peaks_only", "fixed_interval"
    fixed_interval_ms: int = 100
    confidence_threshold: float = 0.1


@dataclass
class DecoderConfig:
    enabled: bool = False
    checkpoint: str = ""          # path to pretrained AE checkpoint (e.g. auto/checkpoints/viable_dense_ae.pt)
    freeze: bool = True           # freeze decoder weights during training


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data_dir: str = "data"
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        for section_name, section_cls in [
            ("video", VideoConfig),
            ("pose", PoseConfig),
            ("flow", FlowConfig),
            ("model", ModelConfig),
            ("training", TrainingConfig),
            ("inference", InferenceConfig),
            ("decoder", DecoderConfig),
        ]:
            if section_name in raw:
                setattr(cfg, section_name, section_cls(**raw[section_name]))

        for key in ("data_dir", "device"):
            if key in raw:
                setattr(cfg, key, raw[key])

        # Derive flattened feature dimensions from canonical factors to avoid
        # stale manual values when max_persons / keypoints change.
        cfg.model.embedding_features = cfg.model.max_persons * cfg.model.embed_dim
        cfg.model.pose_features = cfg.model.max_persons * cfg.pose.n_keypoints * 3
        cfg.model.flow_features = cfg.flow.output_features

        return cfg

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
