"""Unit tests for the dualq pair split, time features and model contract."""

from __future__ import annotations

import math

from pathlib import Path

import pytest
import torch

from taac2026.domain.schema import PCVRSchema
from taac2026.infrastructure.data.batches import PCVRModelInput, PCVRSequenceInput
from taac2026.infrastructure.modeling.time_features import BUCKET_BOUNDARIES
from taac2026.infrastructure.experiments.module_loader import (
    load_experiment_submodule,
    load_module_from_path,
)
from tests.support.model_inputs import dualq_contract_model_input
from tests.support.paths import locate_repo_root

REPO_ROOT = locate_repo_root(Path(__file__))


def _load_module():
    return load_module_from_path(REPO_ROOT / "experiments" / "dualq")


def _load_layers():
    return load_experiment_submodule(REPO_ROOT / "experiments" / "dualq", "layers")


def _load_model_module():
    return load_experiment_submodule(REPO_ROOT / "experiments" / "dualq", "model")


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


def test_pair_split_plan_compiles_offsets_and_slices() -> None:
    layers = _load_layers()
    plan = layers.compile_pair_split(_schema(), [62, 89])

    # Raw offsets index into the full user layout.
    assert plan.user_int_entries == ((1, 0, 1), (87, 3, 1))
    assert plan.pair_int_entries == ((62, 1, 2), (89, 4, 1))
    assert plan.user_dense_entries == ((61, 0, 3),)
    assert plan.pair_dense_entries == ((62, 3, 2), (89, 5, 1))
    assert plan.user_int_reduced == ((1, 0, 1), (87, 1, 1))
    assert plan.user_dense_reduced == ((61, 0, 3),)

    user_int = torch.tensor([[9, 11, 12, 13, 14]], dtype=torch.long)
    user_dense = torch.tensor([[1.0, 2.0, 3.0, 100.0, 200.0, 0.75]], dtype=torch.float32)
    user_int_view, pair_int_view = plan.split_user_int(user_int)
    user_dense_view, pair_dense_view = plan.split_user_dense(user_dense)

    assert user_int_view.tolist() == [[9, 13]]
    assert pair_int_view.tolist() == [[11, 12, 14]]
    assert user_dense_view.tolist() == [[1.0, 2.0, 3.0]]
    # fid 62 (< 89) dense values are log1p-compressed; fid 89 stays raw.
    assert pair_dense_view[0, 0].item() == pytest.approx(math.log1p(100.0), rel=1e-5)
    assert pair_dense_view[0, 1].item() == pytest.approx(math.log1p(200.0), rel=1e-5)
    assert pair_dense_view[0, 2].item() == pytest.approx(0.75)


def test_pair_split_plan_without_pair_columns() -> None:
    layers = _load_layers()
    schema = PCVRSchema(
        format="raw_parquet",
        user_int=[[1, 10, 1], [2, 10, 1]],
        item_int=[[3, 10, 1]],
        user_dense=[[61, 3]],
        item_dense=[],
        seq={},
    )
    plan = layers.compile_pair_split(schema, [62, 89])
    assert plan.pair_int_entries == ()
    assert plan.pair_dense_entries == ()
    user_int = torch.tensor([[9, 8]], dtype=torch.long)
    user_view, pair_view = plan.split_user_int(user_int)
    assert user_view.tolist() == [[9, 8]]
    assert pair_view.shape == (1, 0)


def test_gap_buckets_match_source_semantics() -> None:
    layers = _load_layers()
    timestamps = torch.tensor([[100, 130, 200, 0, 0], [50, 0, 0, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 1], dtype=torch.long)
    buckets = layers.compute_sequence_gap_buckets(timestamps, lengths)

    def expected_gap(gap: int) -> int:
        raw = torch.searchsorted(torch.as_tensor(BUCKET_BOUNDARIES), torch.tensor([gap]))[0]
        return int(raw.clamp(max=len(BUCKET_BOUNDARIES) - 1)) + 1

    assert buckets[0, 0].item() == 0
    assert buckets[0, 1].item() == expected_gap(30)
    assert buckets[0, 2].item() == expected_gap(70)
    assert buckets[0, 3].item() == 0  # padded
    assert buckets[1, 0].item() == 0
    assert buckets[1, 1].item() == 0  # length 1: no consecutive pair


def test_ts_float_and_ts_stat_shapes_and_values() -> None:
    layers = _load_layers()
    timestamps = torch.tensor([[1000, 5000, 0], [0, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([2, 0], dtype=torch.long)
    request = torch.tensor([20000, 20000], dtype=torch.long)

    ts_float = layers.compute_sequence_ts_float(timestamps, lengths, request, "seq_a")
    assert ts_float.shape == (2, 8, 3)
    # Column 0 is log1p(diff_days); padded positions stay zero.
    assert ts_float[0, 0, 0].item() == pytest.approx(math.log1p(19000 / 86400.0))
    assert ts_float[0, 0, 2].item() == 0.0
    assert ts_float[1, 0, :].sum().item() == 0.0
    # Column 1: non-seq_c domain uses d_days directly.
    assert ts_float[0, 1, 1].item() == pytest.approx(15000 / 86400.0)
    # Column 2 is log1p(diff_hours).
    assert ts_float[0, 2, 1].item() == pytest.approx(math.log1p(15000 / 3600.0))

    ts_stat = layers.compute_sequence_ts_stat(timestamps, lengths, request)
    assert ts_stat.shape == (2, 6)
    # log1p of max diff (19000) for row 0; zero row for the empty row.
    assert ts_stat[0, 0].item() == pytest.approx(math.log1p(19000.0))
    assert ts_stat[1, :].sum().item() == 0.0
    # Event-count columns: row 0 diffs are 19000s and 15000s, so only the
    # within-1-day count fires.
    assert ts_stat[0, 3].item() == 0.0
    assert ts_stat[0, 4].item() == 0.0
    assert ts_stat[0, 5].item() == 2.0


def test_global_time_features() -> None:
    layers = _load_layers()
    # 2026-08-06T10:00:00+00:00 UTC = 18:00 local (UTC+8), a Thursday.
    request = torch.tensor([1786000000], dtype=torch.long)
    feats = layers.build_global_time_features(request)
    assert feats.shape == (1, 3)
    hour = int(feats[0, 0].item())
    dow = int(feats[0, 1].item())
    weekend = int(feats[0, 2].item())
    assert 1 <= hour <= 24
    assert 1 <= dow <= 7
    assert weekend in (1, 2)
    assert weekend == (2 if dow >= 6 else 1)


def _small_model(module, *, gradient_checkpointing: bool = False):
    return module.PCVRDualQ(
        schema=_schema(),
        config=module.DualQModelConfig(
            d_model=24,
            emb_dim=8,
            num_queries=6,
            num_blocks=1,
            num_heads=2,
            hidden_mult=2,
            dropout_rate=0.0,
            action_num=1,
            use_time_buckets=True,
            gradient_checkpointing=gradient_checkpointing,
            ns=module.PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        ),
    )


def test_time_aligned_interleave_uses_exact_time_and_puts_padding_last() -> None:
    module = _load_module()
    model = _small_model(module)
    base = dualq_contract_model_input()
    inputs = PCVRModelInput(
        user=base.user,
        item=base.item,
        sequences={
            "seq_a": PCVRSequenceInput(
                values=base.sequences["seq_a"].values,
                lengths=torch.tensor([2, 2], dtype=torch.long),
                timestamps=torch.tensor(
                    [[4998, 4996, 0, 0], [600, 500, 0, 0]], dtype=torch.long
                ),
            ),
            "seq_b": PCVRSequenceInput(
                values=base.sequences["seq_b"].values,
                lengths=torch.tensor([2, 2], dtype=torch.long),
                timestamps=torch.tensor(
                    [[4999, 4997, 0], [200, 100, 0]], dtype=torch.long
                ),
            ),
        },
        request_timestamp=base.request_timestamp,
    )
    time_buckets, gap_buckets, ts_float, ts_stat = model._derive_time_features(inputs)

    _, masks, interleaved_ts, _, _ = model._build_interleaved_seq_tokens(
        inputs,
        time_buckets,
        gap_buckets,
        ts_float,
        ts_stat,
    )

    assert masks[0][0].tolist() == [False, False, False, False, True, True, True]
    ordered_delta_days = torch.expm1(interleaved_ts["interleave"][0, 0, :4])
    ordered_deltas = ordered_delta_days * 86400.0
    assert ordered_deltas.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0], abs=1e-4)


def test_gradient_checkpointing_routes_each_block_through_helper(monkeypatch) -> None:
    module = _load_module()
    model_module = _load_model_module()
    checkpoint_flags: list[bool] = []

    def recording_checkpoint(function, *args, enabled=False, **kwargs):
        checkpoint_flags.append(enabled)
        return function(*args, **kwargs)

    monkeypatch.setattr(model_module, "maybe_gradient_checkpoint", recording_checkpoint)
    model = _small_model(module, gradient_checkpointing=True)

    model(dualq_contract_model_input()).sum().backward()

    assert checkpoint_flags == [True]


def test_dualq_model_forward_backward_and_predict() -> None:
    module = _load_module()
    model = module.PCVRDualQ(
        schema=_schema(),
        config=module.DualQModelConfig(
            d_model=24,
            emb_dim=8,
            num_queries=6,
            num_blocks=1,
            num_heads=2,
            hidden_mult=2,
            dropout_rate=0.0,
            action_num=1,
            use_time_buckets=True,
            gradient_checkpointing=False,
            ns=module.PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        ),
    )

    assert model.num_ns > 0
    assert model.fm_highway is not None
    assert model.fm_highway.output_dim == model.num_item_tokens * (
        model.num_user_tokens + model.num_sequences
    )
    assert model.clsfier[0].in_features == 24 + model.fm_highway.output_dim
    model_input = dualq_contract_model_input()

    logits = model(model_input)
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    loss = logits.sum()
    loss.backward()

    model.eval()
    with torch.no_grad():
        predicted_logits, embeddings = model.predict(model_input)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape == (2, 24)
    assert torch.isfinite(predicted_logits).all()

    # Embeddings land in the sparse parameter group.
    sparse_params = model.get_sparse_params()
    assert sparse_params
    assert all(parameter.requires_grad for parameter in sparse_params)


def test_dualq_hash_compresses_high_cardinality_features() -> None:
    module = _load_module()
    model = module.PCVRDualQ(
        schema=_schema(),
        config=module.DualQModelConfig(
            d_model=24,
            emb_dim=8,
            emb_skip_threshold=4,
            compress_high_cardinality=True,
            num_queries=6,
            num_blocks=1,
            num_heads=2,
            hidden_mult=2,
            dropout_rate=0.0,
            action_num=1,
            ns=module.PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        ),
    )

    assert any(model.user_ns_tokenizer._emb_compressed)
    assert any(model.item_ns_tokenizer._emb_compressed)
    assert all(any(flags) for flags in model._seq_emb_compressed.values())
    assert all(
        embedding.num_embeddings == 5
        for domain in model.raw_seq_domains
        for embedding in model._seq_embs[domain]
    )

    with torch.no_grad():
        logits = model(dualq_contract_model_input())
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_dualq_can_ablate_fm_highway() -> None:
    module = _load_module()
    model = module.PCVRDualQ(
        schema=_schema(),
        config=module.DualQModelConfig(
            d_model=24,
            emb_dim=8,
            num_queries=6,
            num_blocks=1,
            num_heads=2,
            hidden_mult=2,
            dropout_rate=0.0,
            action_num=1,
            use_fm_highway=False,
            ns=module.PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        ),
    )

    assert model.fm_highway is None
    assert model.clsfier[0].in_features == 24
    with torch.no_grad():
        logits = model(dualq_contract_model_input())
    assert logits.shape == (2, 1)


def test_dualq_model_dense_split_and_pair_tokenizer_wired() -> None:
    module = _load_module()
    model = module.PCVRDualQ(
        schema=_schema(),
        config=module.DualQModelConfig(
            d_model=24,
            emb_dim=8,
            num_queries=6,
            num_blocks=1,
            num_heads=2,
            hidden_mult=2,
            dropout_rate=0.0,
            action_num=1,
            use_time_buckets=True,
            gradient_checkpointing=False,
            user_emb_dim=3,  # fid 61 width in the test schema
            user_seq_block_dim=32,
            user_seq_num=10,
            ns=module.PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        ),
    )

    # Schema has no fid 87 user dense block, so the split must fall back to a
    # single dense token and no group splits.
    assert not model._user_dense_split
    assert model.cross_ns_tokenizer is not None  # pair fids 62/89 present
    assert model.num_ns > 0
    with torch.no_grad():
        logits = model(dualq_contract_model_input())
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_dualq_dualq_requires_query_token_sum() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="num_queries == user_q_tokens"):
        module.PCVRDualQ(
            schema=_schema(),
            config=module.DualQModelConfig(
                d_model=24,
                num_queries=4,
                user_q_tokens=3,
                item_q_tokens=2,
                ns=module.PCVRNSConfig(grouping_strategy="singleton"),
            ),
        )


def test_dualq_rejects_pair_fid_in_user_groups() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="pair fid"):
        module.PCVRDualQ(
            schema=_schema(),
            config=module.DualQModelConfig(
                d_model=24,
                num_queries=6,
                ns=module.PCVRNSConfig(
                    grouping_strategy="explicit",
                    user_groups={"U1": [1], "U62": [62]},
                    item_groups={"I1": [3]},
                ),
            ),
        )
