from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from taac2026.application.experiments.registry import load_experiment_package
from taac2026.application.evaluation.runtime import load_train_config
from taac2026.domain.config import PCVR_DATA_CACHE_MODE_CHOICES, PCVRNSConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
from taac2026.infrastructure.data.batches import (
    PCVREntityInput,
    PCVRModelInput,
    PCVRSequenceInput,
)
from taac2026.infrastructure.experiments.module_loader import load_experiment_submodule, load_module_from_path
from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.modeling import safe_key_padding_mask
from tests.support.experiment_matrix import discover_pcvr_experiment_cases, get_experiment_case, load_model_module


EXPERIMENT_CASES = discover_pcvr_experiment_cases()


def _load_package_module(experiment_path: str):
    experiment_case = get_experiment_case(experiment_path)
    return load_module_from_path(experiment_case.package_dir)


def _make_schema() -> PCVRSchema:
    return PCVRSchema(
        format="raw_parquet",
        user_int=[[8, 8, 1], [7, 7, 2]],
        item_int=[[5, 5, 1]],
        user_dense=[[1, 2]],
        item_dense=[[2, 1]],
        seq={
            "seq_a": {"prefix": "seq_a", "ts_fid": 1, "features": [[1, 6], [2, 5], [4, 4]]},
            "seq_b": {"prefix": "seq_b", "ts_fid": 3, "features": [[3, 4], [5, 7]]},
        },
    )


def _make_model(experiment_case, model_module, overrides=None):
    model_class = getattr(model_module, experiment_case.model_class)
    config_kwargs = dict(
        d_model=16,
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
    if overrides:
        config_kwargs.update(overrides)
    experiment = load_experiment_package(experiment_case.path)
    model_config_type = type(experiment.train_defaults.model)

    def build(d_model: int):
        return model_class(
            schema=_make_schema(),
            config=model_config_type(**{**config_kwargs, "d_model": d_model}),
        )

    try:
        return build(16)
    except ValueError as error:
        match = re.search(r"=(\d+)\. Valid T values", str(error))
        if "must be divisible by T" not in str(error) or match is None:
            raise
        return build(int(match.group(1)) * 4)


def _sample_model_input(model_module=None) -> PCVRModelInput:
    user = PCVREntityInput(
        int_values=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long),
        int_missing_mask=torch.zeros(2, 3, dtype=torch.bool),
        dense_values=torch.randn(2, 2),
        dense_missing_mask=torch.zeros(2, 2, dtype=torch.bool),
    )
    item = PCVREntityInput(
        int_values=torch.tensor([[1], [2]], dtype=torch.long),
        int_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
        dense_values=torch.randn(2, 1),
        dense_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
    )
    return PCVRModelInput(
        user=user,
        item=item,
        sequences={
            "seq_a": PCVRSequenceInput(
                values=torch.tensor(
                    [
                        [[1, 2, 0, 0], [2, 3, 0, 0]],
                        [[4, 1, 2, 3], [1, 2, 3, 4]],
                    ],
                    dtype=torch.long,
                ),
                lengths=torch.tensor([2, 4], dtype=torch.long),
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


def _assert_valid_data_pipeline_defaults(data_pipeline: dict[str, object]) -> None:
    cache = data_pipeline["cache"]
    assert isinstance(cache, dict)
    assert cache["mode"] in PCVR_DATA_CACHE_MODE_CHOICES
    assert isinstance(cache["max_batches"], int)
    assert cache["max_batches"] >= 0

    seed = data_pipeline["seed"]
    assert seed is None or isinstance(seed, int)
    assert isinstance(data_pipeline["strict_time_filter"], bool)

    transforms = data_pipeline["transforms"]
    assert isinstance(transforms, list)
    for transform in transforms:
        assert isinstance(transform, dict)
        assert isinstance(transform["name"], str)
        assert isinstance(transform["enabled"], bool)


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_packages_load(experiment_case) -> None:
    experiment = load_experiment_package(experiment_case.path)
    train_defaults = experiment.train_defaults.model_dump(mode="json")

    assert experiment.name == experiment_case.name
    assert experiment.package_dir == experiment_case.package_dir
    assert experiment.train_defaults is not None
    assert experiment.kind == "pcvr"
    assert experiment.model_class_name == experiment_case.model_class
    assert experiment.config_type_name == type(experiment.train_defaults).__name__
    assert train_defaults["model"]["ns"]["grouping_strategy"] == "explicit"
    assert train_defaults["optimizer"]["max_steps"] > 0
    _assert_valid_data_pipeline_defaults(train_defaults["data_pipeline"])
    assert isinstance(train_defaults["model"]["ns"]["user_groups"], dict)
    assert isinstance(train_defaults["model"]["ns"]["item_groups"], dict)
    assert train_defaults["model"]["ns"]["user_groups"]
    assert train_defaults["model"]["ns"]["item_groups"]
    assert "num_hyformer_blocks" not in train_defaults
    assert "symbiosis_use_candidate_decoder" not in train_defaults
    assert "symbiosis_use_field_tokens" not in train_defaults
    assert "symbiosis_recent_tokens" not in train_defaults


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_models_forward_and_predict(experiment_case) -> None:
    model_module = load_model_module(experiment_case)
    if experiment_case.path != "experiments/baseline":
        assert not hasattr(model_module, "PCVRHyFormer")
    assert hasattr(model_module, experiment_case.model_class)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)

    logits = model(model_input)
    loss = logits.sum()
    loss.backward()

    model.eval()
    with torch.no_grad():
        predicted_logits, embeddings = model.predict(model_input)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape[0] == 2
    assert model.num_ns > 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted_logits).all()


def test_symbiosis_train_config_includes_package_specific_model_config() -> None:
    symbiosis_module = _load_package_module("experiments/symbiosis")

    defaults = symbiosis_module.TRAIN_DEFAULTS
    model = defaults.model

    assert isinstance(model, symbiosis_module.SymbiosisModelConfig)
    assert model.v2_use_dense_tokens is True
    assert model.v2_use_missing_tokens is True
    assert model.v3_enabled is True
    assert model.v3_memory_selection_mode == "quality_stratified"
    assert model.ns.grouping_strategy == "explicit"
    assert model.ns.user_tokens == 7
    assert model.ns.item_tokens == 4


def test_symbiosis_train_config_sidecar_round_trips_package_specific_keys(tmp_path: Path) -> None:
    symbiosis_module = _load_package_module("experiments/symbiosis")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_config.json").write_text(
        dumps(build_pcvr_train_config_sidecar(symbiosis_module.TRAIN_DEFAULTS)),
        encoding="utf-8",
    )

    loaded = load_train_config(symbiosis_module.EXPERIMENT.config_type, checkpoint_dir)

    assert isinstance(loaded, symbiosis_module.SymbiosisTrainConfig)
    assert loaded.model.v2_use_dense_tokens is True
    assert loaded.model.v3_memory_selection_mode == "quality_stratified"
    assert loaded.model.ns.user_tokens == 7


def test_symbiosis_v2_tokenizer_outputs_unified_metadata() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)

    batch = model.tokenizer(model_input)

    assert batch.tokens.shape[0] == 2
    assert batch.tokens.shape[1] == model.num_ns + model.num_sequence_tokens
    assert batch.role_ids.shape == (batch.tokens.shape[1],)
    assert batch.domain_ids.shape == (batch.tokens.shape[1],)
    assert batch.risk_ids.shape == (batch.tokens.shape[1],)
    assert batch.cls_index == 0
    assert batch.candidate_index == 1
    assert not hasattr(model, "sequence_memory")
    assert not hasattr(model, "_sparse_tokens")


def test_symbiosis_v2_sparse_missing_changes_unified_tokens() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)
    no_missing_input = replace(
        model_input,
        user=replace(
            model_input.user,
            int_missing_mask=torch.zeros_like(model_input.user.int_values, dtype=torch.bool),
        ),
        item=replace(
            model_input.item,
            int_missing_mask=torch.zeros_like(model_input.item.int_values, dtype=torch.bool),
        ),
    )
    missing_input = replace(
        model_input,
        user=replace(
            model_input.user,
            int_missing_mask=torch.ones_like(model_input.user.int_values, dtype=torch.bool),
        ),
        item=replace(
            model_input.item,
            int_missing_mask=torch.ones_like(model_input.item.int_values, dtype=torch.bool),
        ),
    )

    with torch.no_grad():
        tokens_without_missing = model.tokenizer(no_missing_input).tokens
        tokens_with_missing = model.tokenizer(missing_input).tokens

    assert not torch.allclose(tokens_without_missing, tokens_with_missing)


def test_symbiosis_v2_diagnostics_report_unified_token_health() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)

    model.set_tensorboard_diagnostics_enabled(True)
    model.eval()
    with torch.no_grad():
        model(model_input)

    scalars = model.consume_tensorboard_scalars(phase="train")

    assert "SymbiosisV2/tokens/active_ratio/train" in scalars
    assert "SymbiosisV2/tokens/count/train" in scalars
    assert "SymbiosisV2/tokens/high_risk_ratio/train" in scalars
    assert "SymbiosisV2/embedding/norm_mean/train" in scalars
    assert model.consume_tensorboard_scalars(phase="train") == {}


def test_symbiosis_v2_high_risk_dropout_masks_risk_tokens_during_training() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(
        experiment_case,
        model_module,
        overrides={"v2_high_risk_token_dropout_rate": 1.0},
    )
    model_input = _sample_model_input(model_module)

    model.train()
    batch = model._apply_high_risk_dropout(model.tokenizer(model_input))
    risk_positions = batch.risk_ids > 0

    assert risk_positions.any()
    assert batch.padding_mask[:, risk_positions].all()
    assert torch.equal(batch.tokens[:, risk_positions, :], torch.zeros_like(batch.tokens[:, risk_positions, :]))
    assert not batch.padding_mask[:, batch.cls_index].any()
    assert not batch.padding_mask[:, batch.candidate_index].any()


def test_symbiosis_v2_tokenizer_uses_fixed_event_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(
        experiment_case,
        model_module,
        overrides={
            "v3_enabled": False,
            "v2_recent_event_tokens": 4,
            "v2_memory_event_tokens": 2,
        },
    )
    model_input = _sample_model_input(model_module)
    model_input = replace(
        model_input,
        sequences={
            "seq_a": PCVRSequenceInput(
                values=torch.ones(2, 2, 64, dtype=torch.long),
                lengths=torch.tensor([64, 32], dtype=torch.long),
                timestamps=torch.zeros(2, 64, dtype=torch.long),
            ),
            "seq_b": PCVRSequenceInput(
                values=torch.ones(2, 1, 48, dtype=torch.long),
                lengths=torch.tensor([48, 24], dtype=torch.long),
                timestamps=torch.zeros(2, 48, dtype=torch.long),
            ),
        },
    )
    observed_lengths: list[int] = []

    for tokenizer in model.tokenizer.sequence_tokenizers.values():
        original_forward = tokenizer.forward

        def recording_forward(sequence, timestamps=None, request_timestamp=None, *, _original_forward=original_forward):
            observed_lengths.append(int(sequence.shape[2]))
            return _original_forward(sequence, timestamps, request_timestamp)

        monkeypatch.setattr(tokenizer, "forward", recording_forward)

    model.tokenizer(model_input)

    assert observed_lengths == [6, 6]


def test_symbiosis_v3_tokenizer_uses_source_aware_event_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(
        experiment_case,
        model_module,
        overrides={
            "v3_enabled": True,
            "v3_recent_event_tokens_by_domain": "seq_a:3,seq_b:5",
            "v3_memory_event_tokens_by_domain": "seq_a:2,seq_b:4",
        },
    )
    model_input = _sample_model_input(model_module)
    model_input = replace(
        model_input,
        sequences={
            "seq_a": PCVRSequenceInput(
                values=torch.ones(2, 2, 64, dtype=torch.long),
                lengths=torch.tensor([64, 32], dtype=torch.long),
                timestamps=torch.ones(2, 64, dtype=torch.long),
            ),
            "seq_b": PCVRSequenceInput(
                values=torch.ones(2, 1, 48, dtype=torch.long),
                lengths=torch.tensor([48, 24], dtype=torch.long),
                timestamps=torch.ones(2, 48, dtype=torch.long),
            ),
        },
    )
    observed_lengths: list[int] = []

    for tokenizer in model.tokenizer.sequence_tokenizers.values():
        original_forward = tokenizer.forward

        def recording_forward(sequence, timestamps=None, request_timestamp=None, *, _original_forward=original_forward):
            observed_lengths.append(int(sequence.shape[2]))
            return _original_forward(sequence, timestamps, request_timestamp)

        monkeypatch.setattr(tokenizer, "forward", recording_forward)

    batch = model.tokenizer(model_input)

    assert observed_lengths == [5, 9]
    assert model.num_sequence_tokens == 5 + 9 + model.tokenizer.sequence_stats.num_tokens
    assert batch.tokens.shape[1] == model.num_ns + model.num_sequence_tokens


def test_symbiosis_v2_attention_bias_uses_metadata() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)
    batch = model.tokenizer(model_input)

    mask = model.attention_mask(batch)

    assert mask.shape == (2, 1, batch.tokens.shape[1], batch.tokens.shape[1])
    assert mask[:, :, batch.candidate_index, :].any(dim=-1).all()
    assert mask[:, :, batch.cls_index, :].any(dim=-1).all()


def test_symbiosis_attention_aligns_qk_dtype_for_amp(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    captured_dtypes = None

    def fake_attention(q, k, v, **kwargs):
        nonlocal captured_dtypes
        captured_dtypes = (q.dtype, k.dtype, v.dtype)
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=v.dtype)

    backbone_module = load_experiment_submodule(
        get_experiment_case("experiments/symbiosis").package_dir,
        "backbone",
    )
    monkeypatch.setattr(backbone_module, "scaled_dot_product_attention", fake_attention)
    attention = model_module.UnifiedSelfAttention(d_model=16, num_heads=2, dropout=0.0)
    tokens = torch.randn(2, 5, 16)
    attention_mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        attention(tokens, attention_mask)

    assert captured_dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16)


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_models_backward_with_gradient_checkpointing(experiment_case) -> None:
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module, overrides={"gradient_checkpointing": True})
    model_input = _sample_model_input(model_module)

    logits = model(model_input)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16], ids=["bfloat16", "float16"])
def test_discovered_experiment_models_forward_and_backward_under_amp_autocast(experiment_case, amp_dtype) -> None:
    """Verify every experiment model produces finite logits, loss, and gradients under AMP autocast.

    This test catches NaN issues that only appear in reduced-precision compute
    paths (e.g. overflow in attention softmax, division by zero in normalization,
    or underflow in embedding lookups) that are invisible in the fp32 contract test.
    """
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model.train()
    model_input = _sample_model_input(model_module)
    label = torch.tensor([1.0, 0.0])

    with torch.autocast("cpu", dtype=amp_dtype):
        logits = model(model_input).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label)

    assert logits.shape == (2,), f"expected logits shape (2,), got {logits.shape}"
    assert torch.isfinite(logits).all(), (
        f"Non-finite logits under AMP {amp_dtype}: "
        f"{(~torch.isfinite(logits)).sum()}/{logits.numel()} inf/nan"
    )
    assert torch.isfinite(loss), f"Non-finite loss under AMP {amp_dtype}: {loss.item()}"

    loss.backward()

    # Check that no gradient is NaN/inf after backward.
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad._values() if param.grad.is_sparse else param.grad
            assert torch.isfinite(grad).all(), (
                f"Non-finite gradient for {name} under AMP {amp_dtype}: "
                f"{(~torch.isfinite(grad)).sum()}/{grad.numel()} inf/nan"
            )


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16], ids=["bfloat16", "float16"])
def test_discovered_experiment_models_predict_under_amp_autocast(experiment_case, amp_dtype) -> None:
    """Verify predict path produces finite logits and embeddings under AMP autocast."""
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model.eval()
    model_input = _sample_model_input(model_module)

    with torch.no_grad(), torch.autocast("cpu", dtype=amp_dtype):
        predicted_logits, embeddings = model.predict(model_input)

    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape[0] == 2
    assert torch.isfinite(predicted_logits).all(), (
        f"Non-finite predict logits under AMP {amp_dtype}: "
        f"{(~torch.isfinite(predicted_logits)).sum()}/{predicted_logits.numel()} inf/nan"
    )
    assert torch.isfinite(embeddings).all(), (
        f"Non-finite embeddings under AMP {amp_dtype}: "
        f"{(~torch.isfinite(embeddings)).sum()}/{embeddings.numel()} inf/nan"
    )


def test_symbiosis_keeps_sequence_width_stable_for_compile() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(
        experiment_case,
        model_module,
        overrides={
            "v3_enabled": False,
            "v2_recent_event_tokens": 4,
            "v2_memory_event_tokens": 2,
        },
    )
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
                values=torch.tensor(
                    [
                        [[1, 2, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0]],
                        [[4, 1, 2, 0, 0, 0], [1, 2, 3, 0, 0, 0]],
                    ],
                    dtype=torch.long,
                ),
                lengths=torch.tensor([2, 3], dtype=torch.long),
                timestamps=torch.zeros(2, 6, dtype=torch.long),
            ),
            "seq_b": PCVRSequenceInput(
                values=torch.tensor([[[1, 0, 0, 0, 0]], [[2, 3, 0, 0, 0]]], dtype=torch.long),
                lengths=torch.tensor([1, 2], dtype=torch.long),
                timestamps=torch.zeros(2, 5, dtype=torch.long),
            ),
        },
        request_timestamp=torch.tensor([1000, 1000], dtype=torch.long),
    )

    batch = model.tokenizer(model_input)

    expected_sequence_tokens = (4 + 2) * len(model.tokenizer.seq_domains) + model.tokenizer.sequence_stats.num_tokens
    assert model.num_sequence_tokens == expected_sequence_tokens
    assert batch.tokens.shape[1] == model.num_ns + expected_sequence_tokens


def test_safe_key_padding_mask_unmasks_first_position_for_fully_padded_rows() -> None:
    mask = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )

    safe_mask = safe_key_padding_mask(mask)

    assert torch.equal(
        safe_mask,
        torch.tensor(
            [
                [False, True, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        ),
    )


def test_symbiosis_compile_handles_multiple_sequence_lengths() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    input_a = _sample_model_input(model_module)
    input_b = replace(
        input_a,
        sequences={
            "seq_a": PCVRSequenceInput(
                values=input_a.sequences["seq_a"].values,
                lengths=torch.tensor([1, 2], dtype=torch.long),
                timestamps=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            ),
            "seq_b": PCVRSequenceInput(
                values=input_a.sequences["seq_b"].values,
                lengths=torch.tensor([3, 1], dtype=torch.long),
                timestamps=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
            ),
        },
    )

    with torch.no_grad():
        eager_logits_a = model(input_a)
        eager_logits_b = model(input_b)

    compiled = torch.compile(model, backend="eager")
    with torch.no_grad():
        logits_a = compiled(input_a)
        logits_b = compiled(input_b)

    assert logits_a.shape == (2, 1)
    assert logits_b.shape == (2, 1)
    assert torch.allclose(logits_a, eager_logits_a)
    assert torch.allclose(logits_b, eager_logits_b)


def test_symbiosis_unified_stage_uses_compact_context_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)
    observed_unified_token_counts: list[int] = []
    original_forward = model.blocks[0].forward

    def recording_forward(tokens, padding_mask, attention_mask=None):
        observed_unified_token_counts.append(int(tokens.shape[1]))
        return original_forward(tokens, padding_mask, attention_mask)

    monkeypatch.setattr(model.blocks[0], "forward", recording_forward)

    logits = model(model_input)

    context_token_count = model.tokenizer(model_input).tokens.shape[1]

    assert logits.shape == (2, 1)
    assert observed_unified_token_counts == [context_token_count]
    assert context_token_count == model.num_ns + model.num_sequence_tokens
    assert model.num_sequence_tokens == model.tokenizer.num_sequence_tokens


def test_symbiosis_v2_flags_disable_optional_modules() -> None:
    experiment_case = get_experiment_case("experiments/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(
        experiment_case,
        model_module,
        overrides={
            "v2_use_dense_tokens": False,
            "v2_use_missing_tokens": False,
            "v2_use_sequence_stats_tokens": False,
            "v2_use_metadata_attention_bias": False,
            "v2_use_candidate_readout": False,
            "v2_compile_backbone": False,
        },
    )
    model_input = _sample_model_input(model_module)

    assert model.tokenizer.use_dense_tokens is False
    assert model.tokenizer.use_missing_tokens is False
    assert model.tokenizer.use_sequence_stats_tokens is False
    assert model.attention_mask.enabled is False
    assert model.pooler.use_candidate_readout is False
    assert model.symbiosis_v2_compile_backbone is False

    logits = model(model_input)
    predicted_logits, embeddings = model.predict(model_input)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape == (2, model.pooler.output_dim)
    assert torch.isfinite(logits).all()
