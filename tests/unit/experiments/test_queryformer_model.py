"""Unit tests for QueryFormer tokenization and interaction contracts."""

from __future__ import annotations

from pathlib import Path

import torch

from taac2026.domain.config import PCVRNSConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.infrastructure.data.batches import (
    PCVREntityInput,
    PCVRModelInput,
    PCVRSequenceInput,
)
from taac2026.infrastructure.experiments.module_loader import (
    load_experiment_submodule,
    load_module_from_path,
)
from tests.support.paths import locate_repo_root

REPO_ROOT = locate_repo_root(Path(__file__))
PACKAGE_DIR = REPO_ROOT / "experiments" / "queryformer"


def _package():
    return load_module_from_path(PACKAGE_DIR)


def _model_module():
    return load_experiment_submodule(PACKAGE_DIR, "model")


def _schema() -> PCVRSchema:
    return PCVRSchema(
        format="raw_parquet",
        user_int=[[1, 100, 1], [2, 3, 2]],
        item_int=[[3, 8, 1]],
        user_dense=[[61, 2], [87, 1]],
        item_dense=[[124, 2]],
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


def _inputs() -> PCVRModelInput:
    return PCVRModelInput(
        user=PCVREntityInput(
            int_values=torch.tensor([[99, 1, 2], [75, 2, 3]], dtype=torch.long),
            int_missing_mask=torch.zeros(2, 3, dtype=torch.bool),
            dense_values=torch.randn(2, 3),
            dense_missing_mask=torch.zeros(2, 3, dtype=torch.bool),
        ),
        item=PCVREntityInput(
            int_values=torch.tensor([[1], [2]], dtype=torch.long),
            int_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
            dense_values=torch.randn(2, 2),
            dense_missing_mask=torch.zeros(2, 2, dtype=torch.bool),
        ),
        sequences={
            "seq_a": PCVRSequenceInput(
                values=torch.tensor(
                    [[[8, 4, 0]], [[7, 3, 2]]],
                    dtype=torch.long,
                ),
                lengths=torch.tensor([2, 3], dtype=torch.long),
                timestamps=torch.tensor(
                    [[900, 800, 0], [900, 800, 700]], dtype=torch.long
                ),
            ),
            "seq_b": PCVRSequenceInput(
                values=torch.tensor([[[60, 0]], [[50, 40]]], dtype=torch.long),
                lengths=torch.tensor([1, 2], dtype=torch.long),
                timestamps=torch.tensor([[700, 0], [600, 500]], dtype=torch.long),
            ),
        },
        request_timestamp=torch.tensor([1_000, 1_000], dtype=torch.long),
    )


def _small_config(package, **updates):
    values = {
        "d_model": 16,
        "emb_dim": 8,
        "num_queries": 2,
        "num_blocks": 1,
        "num_heads": 2,
        "hidden_mult": 2,
        "dropout_rate": 0.0,
        "action_num": 1,
        "use_time_buckets": True,
        "emb_skip_threshold": 4,
        "num_embedding_columns": 2,
        "dcn_num_layers": 2,
        "compress_high_cardinality": True,
        "ns": PCVRNSConfig(grouping_strategy="singleton"),
    }
    values.update(updates)
    return package.QueryFormerModelConfig(**values)


def test_dense_field_tokenizer_preserves_schema_boundaries() -> None:
    tokenizer = (
        _model_module()
        .DenseFieldTokenizer([2, 1], d_model=8, num_cross_layers=2, num_columns=2)
        .eval()
    )
    values = torch.tensor([[1.0, 2.0, 3.0]])
    missing = torch.zeros_like(values, dtype=torch.bool)

    baseline = tokenizer(values, missing)
    changed = tokenizer(torch.tensor([[1.0, 2.0, 30.0]]), missing)

    assert baseline.shape == (1, 2, 2, 8)
    assert torch.equal(baseline[:, :, 0], changed[:, :, 0])
    assert not torch.equal(baseline[:, :, 1], changed[:, :, 1])


def test_columns_use_disjoint_ranges_in_one_compressed_embedding_table() -> None:
    package = _package()
    model = package.PCVRQueryFormer(schema=_schema(), config=_small_config(package))

    embedding = model.user_sparse.bank.compressed_embeddings[0]
    column_ids = embedding.column_ids(torch.tensor([[1, 4]]))

    assert model.num_embedding_columns == 2
    assert embedding.weight.shape == (9, 8)
    assert column_ids.tolist() == [[[1, 4], [5, 8]]]
    assert not torch.equal(embedding.weight[1:5], embedding.weight[5:9])
    assert model.num_ns == 12


def test_sparse_token_counts_follow_configured_groups() -> None:
    package = _package()
    config = _small_config(
        package,
        num_embedding_columns=1,
        ns=PCVRNSConfig(
            grouping_strategy="explicit",
            user_groups={"user": [1, 2]},
            item_groups={"item": [3]},
        ),
    )

    model = package.PCVRQueryFormer(schema=_schema(), config=config)

    # One user sparse group + two user dense fields + one item sparse group
    # + one item dense field.
    assert model.num_ns == 5


def test_queryformer_high_cardinality_forward_backward_and_predict() -> None:
    package = _package()
    model = package.PCVRQueryFormer(schema=_schema(), config=_small_config(package))
    inputs = _inputs()

    logits = model(inputs)
    logits.sum().backward()

    compressed = model.user_sparse.bank.compressed_embeddings[0]
    touched_rows = compressed.weight.grad._indices()[0]

    model.eval()
    with torch.no_grad():
        predicted_logits, embeddings = model.predict(inputs)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape == (2, 16)
    assert torch.isfinite(logits).all()
    assert ((1 <= touched_rows) & (touched_rows <= 4)).any()
    assert ((5 <= touched_rows) & (touched_rows <= 8)).any()
    assert model.get_sparse_params()
    assert model.get_dense_params()


def test_column_linear_uses_independent_weights_without_a_python_column_loop() -> None:
    linear = _model_module().ColumnLinear(2, 2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[[1.0], [0.0]], [[0.0], [2.0]]]))
        linear.bias.zero_()

    output = linear(torch.tensor([[[3.0, 4.0], [3.0, 4.0]]]))

    assert output.tolist() == [[[3.0], [8.0]]]


def test_queryformer_torch_compile_preserves_forward() -> None:
    package = _package()
    model = package.PCVRQueryFormer(
        schema=_schema(), config=_small_config(package)
    ).eval()
    inputs = _inputs()

    with torch.no_grad():
        eager = model(inputs)
        compiled = torch.compile(model, backend="eager")(inputs)

    assert torch.allclose(compiled, eager)


def test_queryformer_runtime_compile_targets_dense_backbone(monkeypatch) -> None:
    package = _package()
    model = package.PCVRQueryFormer(
        schema=_schema(), config=_small_config(package)
    ).eval()
    compile_calls = []

    def record_compile(callable_obj):
        compile_calls.append(callable_obj)
        return callable_obj

    monkeypatch.setattr(torch, "compile", record_compile)
    model.prepare_for_runtime_compile()
    model.prepare_for_runtime_compile()

    assert model.uses_internal_compile is True
    assert compile_calls == [model._run_backbone]
    assert model(_inputs()).shape == (2, 1)


def test_all_four_attention_modules_can_be_ablated_together() -> None:
    package = _package()
    model = package.PCVRQueryFormer(
        schema=_schema(),
        config=_small_config(
            package,
            use_query_self_attention=False,
            use_query_cross_attention=False,
            use_query_seq_cross_attention=False,
            use_seq_query_cross_attention=False,
        ),
    )

    logits = model(_inputs())
    block = model.co_blocks[0]
    bridge = model.sequence_bridges["seq_a"]

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    assert block.user_self is None
    assert block.user_from_item is None
    assert bridge.seq_query_attention is None
    assert bridge.query_seq_attention is None
    assert bridge.mlp_query_in is not None
    assert bridge.mlp_query_out is not None
