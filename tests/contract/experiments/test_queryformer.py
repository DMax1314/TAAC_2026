"""Contract tests for the article-derived QueryFormer experiment."""

from __future__ import annotations

from pathlib import Path

from taac2026.application.evaluation.runtime import load_train_config
from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
from taac2026.infrastructure.experiments.module_loader import load_module_from_path
from taac2026.infrastructure.io.json import dumps
from tests.support.paths import locate_repo_root

REPO_ROOT = locate_repo_root(Path(__file__))
PACKAGE_DIR = REPO_ROOT / "experiments" / "queryformer"


def _package():
    return load_module_from_path(PACKAGE_DIR)


def test_queryformer_package_declares_typed_experiment() -> None:
    package = _package()
    experiment = package.EXPERIMENT

    assert experiment.name == "pcvr_queryformer"
    assert experiment.model_class_name == "PCVRQueryFormer"
    assert experiment.package_dir == PACKAGE_DIR.resolve()
    assert experiment.config_type is package.QueryFormerTrainConfig
    assert isinstance(experiment.train_defaults.model, package.QueryFormerModelConfig)


def test_queryformer_defaults_match_published_champion_components() -> None:
    defaults = _package().EXPERIMENT.train_defaults
    model = defaults.model

    assert model.num_embedding_columns == 4
    assert model.dcn_num_layers == 2
    assert model.compress_high_cardinality is True
    assert model.use_query_self_attention is True
    assert model.use_query_cross_attention is True
    assert model.use_query_seq_cross_attention is True
    assert model.use_seq_query_cross_attention is True
    assert defaults.optimizer.dense_optimizer_type == "muon"
    assert defaults.optimizer.scheduler_type == "cosine"
    assert defaults.ema.enabled is True
    assert defaults.runtime.amp is True
    assert defaults.runtime.amp_dtype == "bfloat16"
    assert defaults.runtime.compile is True


def test_queryformer_sidecar_round_trips_architecture_switches(
    tmp_path: Path,
) -> None:
    package = _package()
    experiment = package.EXPERIMENT
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_config.json").write_text(
        dumps(build_pcvr_train_config_sidecar(experiment.train_defaults)),
        encoding="utf-8",
    )

    loaded = load_train_config(experiment.config_type, checkpoint_dir)

    assert isinstance(loaded, package.QueryFormerTrainConfig)
    assert isinstance(loaded.model, package.QueryFormerModelConfig)
    assert loaded.model.num_embedding_columns == 4
    assert loaded.model.dcn_num_layers == 2
    assert loaded.model.use_seq_query_cross_attention is True
    assert loaded.model.use_query_seq_cross_attention is True
