"""Contract tests for the dualq experiment package.

Covers the discovery/sidecar/typed-config chain end to end:
``schema + typed config -> PCVRDualQ -> PCVRModelInput -> forward/predict``
plus train-config subclass reconstruction from the checkpoint sidecar.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from taac2026.domain.config import PCVRNSConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
from taac2026.infrastructure.experiments.module_loader import load_module_from_path
from taac2026.infrastructure.io.json import dumps
from tests.support.model_inputs import dualq_contract_model_input
from tests.support.paths import locate_repo_root

REPO_ROOT = locate_repo_root(Path(__file__))
PACKAGE_DIR = REPO_ROOT / "experiments" / "dualq"


def _package():
    return load_module_from_path(PACKAGE_DIR)


def _schema() -> PCVRSchema:
    return PCVRSchema(
        format="raw_parquet",
        user_int=[[1, 10, 1], [62, 20, 2], [87, 5, 1], [89, 30, 1]],
        item_int=[[3, 10, 1]],
        user_dense=[[61, 3], [62, 2], [89, 1]],
        item_dense=[[124, 3]],
        seq={
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 9,
                "features": [[9, 128], [10, 16]],
            },
            "seq_b": {
                "prefix": "domain_b_seq",
                "ts_fid": 27,
                "features": [[27, 64], [28, 8]],
            },
        },
    )


def test_dualq_package_declares_typed_experiment() -> None:
    package = _package()
    experiment = package.EXPERIMENT

    assert experiment.name == "pcvr_dualq"
    assert experiment.model_class_name == "PCVRDualQ"
    assert experiment.package_dir == PACKAGE_DIR.resolve()
    assert experiment.config_type is package.DualQTrainConfig
    assert experiment.metadata["config_type"] == "DualQTrainConfig"
    assert isinstance(experiment.train_defaults, package.DualQTrainConfig)
    assert isinstance(experiment.train_defaults.model, package.DualQModelConfig)


def test_dualq_train_defaults_cover_source_switches() -> None:
    package = _package()
    model = package.EXPERIMENT.train_defaults.model

    assert model.d_model == 192
    assert model.num_queries == 6
    assert model.user_q_tokens == 4
    assert model.item_q_tokens == 2
    assert model.seq_interest_ratios == "1.0,0.7"
    assert model.seq_interest_ratio_list == [1.0, 0.7]
    assert model.pair_feature_fids == "62,63,64,65,66,89,90,91"
    assert model.use_global_time_token is True
    assert model.use_seq_gap_buckets is True
    assert model.use_time_gap_domain_gates is True
    assert model.use_fid87_token_residual is True
    assert model.use_time_decay_summary is True
    assert model.use_time_aligned_interleave is True
    assert package.EXPERIMENT.train_defaults.data.split_strategy == "timestamp_auto"
    assert model.ns.grouping_strategy == "explicit"
    assert model.ns.tokenizer_type == "rankmixer"
    assert model.ns.user_tokens == 14
    assert model.ns.item_tokens == 3
    assert model.ns.user_groups
    assert model.ns.item_groups


def test_dualq_train_config_sidecar_round_trips_package_specific_keys(tmp_path: Path) -> None:
    package = _package()
    experiment = package.EXPERIMENT
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_config.json").write_text(
        dumps(build_pcvr_train_config_sidecar(experiment.train_defaults)),
        encoding="utf-8",
    )

    loaded = experiment._load_train_config(checkpoint_dir)

    assert isinstance(loaded, package.DualQTrainConfig)
    assert isinstance(loaded.model, package.DualQModelConfig)
    assert loaded.model.d_model == 192
    assert loaded.model.user_q_tokens == 4
    assert loaded.model.item_q_tokens == 2
    assert loaded.model.seq_interest_ratios == "1.0,0.7"
    assert loaded.model.pair_feature_fids == "62,63,64,65,66,89,90,91"
    assert loaded.model.use_time_decay_summary is True
    assert loaded.model.ns.user_tokens == 14
    assert loaded.model.ns.item_tokens == 3
    assert loaded.model.ns.user_groups == experiment.train_defaults.model.ns.user_groups


def test_dualq_model_contract_forward_and_predict() -> None:
    package = _package()
    experiment = package.EXPERIMENT
    # TRAIN_DEFAULTS with a small synthetic schema: override only the ns
    # grouping and the dense dims so the schema stays tiny; every dualq
    # switch (pair fids, global time token, gap buckets, interleave, ...)
    # comes from the real defaults.
    model_config = experiment.train_defaults.model.model_copy(
        update={
            "user_emb_dim": 3,
            "ns": PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        }
    )
    model = experiment.model_type(schema=_schema(), config=model_config)
    model_input = dualq_contract_model_input()

    logits = model(model_input)
    assert logits.shape == (2, 1)
    loss = logits.sum()
    loss.backward()

    model.eval()
    with torch.no_grad():
        predicted_logits, embeddings = model.predict(model_input)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape == (2, 192)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted_logits).all()


def test_dualq_model_rejects_pair_fid_in_user_groups() -> None:
    package = _package()
    with pytest.raises(ValueError, match="pair fid"):
        package.EXPERIMENT.model_type(
            schema=_schema(),
            config=package.DualQModelConfig(
                d_model=192,
                num_queries=6,
                ns=PCVRNSConfig(
                    grouping_strategy="explicit",
                    user_groups={"U1": [1], "U62": [62]},
                    item_groups={"I1": [3]},
                ),
            ),
        )
