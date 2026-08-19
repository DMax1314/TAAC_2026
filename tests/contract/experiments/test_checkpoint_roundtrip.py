"""End-to-end checkpoint round-trip: typed config -> sidecars -> model rebuild -> forward.

Covers the full consumption chain of a training product: schema sidecar,
typed train_config sidecar, config rebuild via ``experiment.config_type``,
uniform model construction from ``schema + config.model``, and a forward pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from taac2026.application.evaluation.runtime import load_runtime_schema, load_train_config
from taac2026.domain.config import PCVRNSConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.domain.sidecar import load_pcvr_train_config_sidecar
from taac2026.infrastructure.checkpoints import write_checkpoint_sidecars
from taac2026.infrastructure.data.batches import PCVREntityInput, PCVRModelInput, PCVRSequenceInput
from taac2026.infrastructure.experiments.module_loader import load_module_from_path
from taac2026.infrastructure.io.json import dumps, read_path
from tests.support.experiment_matrix import get_experiment_case


@pytest.mark.parametrize(
    "experiment_path",
    ["experiments/baseline", "experiments/symbiosis", "experiments/dualq"],
)
def test_checkpoint_roundtrip_rebuilds_model_and_predicts(
    experiment_path: str,
    tmp_path: Path,
) -> None:
    experiment = load_module_from_path(get_experiment_case(experiment_path).package_dir).EXPERIMENT

    schema = PCVRSchema(
        format="raw_parquet",
        user_int=[[8, 8, 1], [7, 7, 2]],
        item_int=[[5, 5, 1]],
        user_dense=[[1, 2]],
        item_dense=[[2, 1]],
        seq={
            "seq_a": {"prefix": "seq_a", "ts_fid": 1, "features": [[1, 6], [2, 5]]},
            "seq_b": {"prefix": "seq_b", "ts_fid": 3, "features": [[3, 4], [5, 7]]},
        },
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(dumps(schema.model_dump(mode="json")), encoding="utf-8")

    model_config_type = type(experiment.train_defaults.model)
    base_kwargs = dict(
        emb_dim=8,
        num_blocks=1,
        num_heads=2,
        hidden_mult=2,
        dropout_rate=0.0,
        action_num=1,
        use_time_buckets=False,
        gradient_checkpointing=False,
        ns=PCVRNSConfig(
            grouping_strategy="singleton",
            tokenizer_type="rankmixer",
            user_tokens=2,
            item_tokens=1,
        ),
    )

    def build(d_model: int):
        return experiment.config_type(
            model=model_config_type(**{**base_kwargs, "d_model": d_model})
        )

    def build_model(d_model: int):
        return experiment.model_type(schema=schema, config=build(d_model).model)

    try:
        model = build_model(16)
        config = experiment.config_type(model=model_config_type(**{**base_kwargs, "d_model": 16}))
    except ValueError as error:
        match = re.search(r"=(\d+)\. Valid T values", str(error))
        assert match is not None, str(error)
        d_model = int(match.group(1)) * 4
        model = build_model(d_model)
        config = experiment.config_type(model=model_config_type(**{**base_kwargs, "d_model": d_model}))
    del model

    checkpoint_dir = tmp_path / "global_step1"
    written = write_checkpoint_sidecars(
        checkpoint_dir,
        schema_path=schema_path,
        train_config=config,
    )
    assert set(written) == {"schema", "train_config"}

    # 1. schema sidecar survived.
    schema_payload = read_path(checkpoint_dir / "schema.json")
    rebuilt_schema = PCVRSchema.model_validate(schema_payload)
    assert rebuilt_schema.model_dump(mode="json") == schema.model_dump(mode="json")

    # 2. typed train_config sidecar survives and rebuilds via experiment.config_type.
    sidecar_payload = read_path(checkpoint_dir / "train_config.json")
    rebuilt_config = load_pcvr_train_config_sidecar(
        sidecar_payload,
        config_type=experiment.config_type,
    )
    assert type(rebuilt_config) is experiment.config_type
    assert rebuilt_config.model_dump() == config.model_dump()
    assert rebuilt_config.model.d_model == config.model.d_model

    # 3. evaluation runtime rebuilds the same config and resolves the checkpoint schema.
    assert load_train_config(experiment.config_type, checkpoint_dir).model_dump() == config.model_dump()
    resolved_schema_path, _schema_payload = load_runtime_schema(
        schema_path=None,
        checkpoint_dir=checkpoint_dir,
        mode="evaluation",
    )
    assert resolved_schema_path == (checkpoint_dir / "schema.json").resolve()

    # 4. uniform model construction from schema + config.model, then forward/predict.
    model = experiment.model_type(schema=rebuilt_schema, config=rebuilt_config.model).eval()

    model_input = PCVRModelInput(
        user=PCVREntityInput(
            int_values=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long),
            int_missing_mask=torch.zeros(2, 3, dtype=torch.bool),
            dense_values=torch.randn(2, 2),
            dense_missing_mask=torch.zeros(2, 2, dtype=torch.bool),
        ),
        item=PCVREntityInput(
            int_values=torch.tensor([[1], [2]], dtype=torch.long),
            int_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
            dense_values=torch.randn(2, 1),
            dense_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
        ),
        sequences={
            "seq_a": PCVRSequenceInput(
                values=torch.tensor([[[1, 0, 0, 0]], [[2, 3, 0, 0]]], dtype=torch.long),
                lengths=torch.tensor([1, 2], dtype=torch.long),
                timestamps=torch.zeros(2, 4, dtype=torch.long),
            ),
            "seq_b": PCVRSequenceInput(
                values=torch.tensor([[[1, 0, 0]], [[2, 3, 0]]], dtype=torch.long),
                lengths=torch.tensor([1, 2], dtype=torch.long),
                timestamps=torch.zeros(2, 3, dtype=torch.long),
            ),
        },
        request_timestamp=torch.tensor([1000, 1000], dtype=torch.long),
    )
    with torch.no_grad():
        logits = model(model_input)
        predicted_logits, _embeddings = model.predict(model_input)
    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted_logits).all()
