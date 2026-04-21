import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_splits import (
    DEFAULT_DISPOSITION_MODEL,
    collect_split_candidates,
    resolve_split_target,
)
from src.config import Config
from src.data.embeddings import save_embeddings_artifacts


def _write_labels(scene_dir: Path, values: np.ndarray) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "labels.npy", values.astype(np.float32))


def _write_embeddings(scene_dir: Path, model_name: str, max_persons: int) -> None:
    emb_path = scene_dir / "embeddings" / f"{Path(model_name).stem}.npy"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((32, max_persons, 512), dtype=np.float32)
    save_embeddings_artifacts(emb_path, data, max_persons=max_persons)


def test_resolve_split_target_for_disposition_uses_disposition_outputs() -> None:
    cfg = Config()

    target = resolve_split_target("disposition", cfg)

    assert target.train_filename == "disposition_train.json"
    assert target.val_filename == "disposition_val.json"
    assert target.model_name == DEFAULT_DISPOSITION_MODEL
    assert target.strict_model_embeddings is True


def test_collect_split_candidates_for_disposition_requires_matching_embeddings(tmp_path: Path) -> None:
    cfg = Config()
    processed_dir = tmp_path / "processed"

    matching_scene = processed_dir / "scene_matching"
    _write_labels(matching_scene, np.linspace(0.0, 1.0, 240))
    _write_embeddings(matching_scene, DEFAULT_DISPOSITION_MODEL, cfg.model.max_persons)

    wrong_model_scene = processed_dir / "scene_wrong_model"
    _write_labels(wrong_model_scene, np.linspace(0.0, 1.0, 240))
    _write_embeddings(wrong_model_scene, cfg.pose.model_name, cfg.model.max_persons)

    legacy_scene = processed_dir / "scene_legacy"
    _write_labels(legacy_scene, np.linspace(0.0, 1.0, 240))
    legacy_embeddings = np.zeros((32, cfg.model.max_persons, 512), dtype=np.float32)
    save_embeddings_artifacts(
        legacy_scene / "embeddings.npy",
        legacy_embeddings,
        max_persons=cfg.model.max_persons,
    )

    candidate_ids, stats, skipped = collect_split_candidates(
        processed_dir,
        model_name=DEFAULT_DISPOSITION_MODEL,
        max_persons=cfg.model.max_persons,
        require_embeddings=True,
        strict_model_embeddings=True,
    )

    assert candidate_ids == ["scene_matching"]
    assert "scene_matching" in stats
    assert (
        "scene_wrong_model",
        "missing model-specific embeddings at vrlens-finetunes-multiclass-v2-yolo26m-pose.npy",
    ) in skipped
    assert (
        "scene_legacy",
        "missing model-specific embeddings at vrlens-finetunes-multiclass-v2-yolo26m-pose.npy",
    ) in skipped