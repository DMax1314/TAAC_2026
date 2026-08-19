from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVREMAConfig,
    PCVRFeatureMaskConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRModelConfig,
    PCVROptimizerConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)
from taac2026.domain.runtime_config import RuntimeExecutionConfig
import taac2026.application.training.cli as training_cli
from taac2026.application.training.cli import main, parse_train_args
from taac2026.domain.experiment import FunctionExperiment
from taac2026.infrastructure.experiments.module_loader import load_module_from_path
from taac2026.infrastructure.io.json import loads
import taac2026.application.training.workflow as workflow_module
from taac2026.application.training.workflow import PCVRTrainDataBundle, build_train_model
from taac2026.application.training.args import parse_pcvr_train_config


REPO_ROOT = Path(__file__).resolve().parents[4]


def _minimal_experiment(*, requires_dataset: bool, kind: str = "maintenance") -> FunctionExperiment:
    def train(request):
        return {
            "dataset_path": None if request.dataset_path is None else str(request.dataset_path),
            "run_dir": str(request.run_dir),
        }

    return FunctionExperiment(
        name="minimal_experiment",
        kind=kind,
        requires_dataset=requires_dataset,
        train_fn=train,
    )


def test_parse_train_args_forwards_experiment_specific_options() -> None:
    args, extra = parse_train_args(
        [
            "--experiment",
            "experiments/baseline",
            "--dataset-path",
            "/data/train",
            "--schema-path",
            "/data/schema.json",
            "--batch_size",
            "8",
        ]
    )

    assert args.experiment == "experiments/baseline"
    assert args.dataset_path == "/data/train"
    assert args.schema_path == "/data/schema.json"
    assert extra == ["--batch_size", "8"]


def test_parse_train_args_allows_missing_dataset_path() -> None:
    args, extra = parse_train_args(["--experiment", "experiments/host_device_info"])

    assert args.experiment == "experiments/host_device_info"
    assert args.dataset_path is None
    assert extra == []


def test_parse_train_args_requires_experiment() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(["--dataset-path", "/data/train"])


def test_training_main_allows_experiment_without_dataset_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiments" / "maintenance" / "maintenance_exp"
    experiment = _minimal_experiment(requires_dataset=False)
    monkeypatch.setattr(training_cli, "load_experiment_package", lambda _path: experiment)
    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--run-dir",
        str(tmp_path / "outputs"),
        "--json",
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert "\n" not in captured.out.strip()
    assert payload["dataset_path"] is None
    assert payload["run_dir"] == str(tmp_path / "outputs")


def test_training_main_rejects_missing_dataset_for_dataset_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment_dir = tmp_path / "experiments" / "maintenance" / "dataset_exp"
    monkeypatch.setattr(training_cli, "load_experiment_package", lambda _path: _minimal_experiment(requires_dataset=True))

    with pytest.raises(ValueError, match="requires --dataset-path"):
        main(["--experiment", str(experiment_dir), "--run-dir", str(tmp_path / "outputs")])


def test_training_main_allows_missing_dataset_for_pcvr_kind_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiments" / "pcvr" / "pcvr_exp"
    monkeypatch.setattr(training_cli, "load_experiment_package", lambda _path: _minimal_experiment(requires_dataset=True, kind="pcvr"))

    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--run-dir",
        str(tmp_path / "outputs"),
        "--json",
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert payload["dataset_path"] is None


def test_training_main_allows_explicit_dataset_for_local_pcvr_kind_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiments" / "pcvr" / "pcvr_exp"
    monkeypatch.setattr(training_cli, "load_experiment_package", lambda _path: _minimal_experiment(requires_dataset=True, kind="pcvr"))

    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--dataset-path",
        "/tmp/custom.parquet",
        "--json",
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert payload["dataset_path"] == "/tmp/custom.parquet"


def test_training_main_allows_explicit_dataset_for_bundle_pcvr_kind_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiments" / "pcvr" / "pcvr_exp"
    monkeypatch.setattr(training_cli, "load_experiment_package", lambda _path: _minimal_experiment(requires_dataset=True, kind="pcvr"))
    monkeypatch.setenv("TAAC_BUNDLE_MODE", "1")

    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--dataset-path",
        "/tmp/custom.parquet",
        "--json",
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert payload["dataset_path"] == "/tmp/custom.parquet"


def test_parse_pcvr_train_config_accepts_runtime_flags(tmp_path: Path) -> None:
    config = parse_pcvr_train_config(
        [
            "--runtime.amp",
            "--runtime.amp_dtype",
            "float16",
            "--runtime.no_compile",
            "--runtime.no_deterministic",
            "--runtime.progress_log_interval_steps",
            "25",
            "--ema.enabled",
            "--ema.decay",
            "0.99",
            "--ema.start_step",
            "10",
            "--ema.update_every_n_steps",
            "2",
            "--model.gradient_checkpointing",
        ],
        config_type=PCVRTrainConfig,
        defaults=PCVRTrainConfig(),
    )

    assert config.runtime.amp is True
    assert config.runtime.amp_dtype == "float16"
    assert config.runtime.compile is False
    assert config.runtime.deterministic is False
    assert config.runtime.progress_log_interval_steps == 25
    assert config.ema.enabled is True
    assert config.ema.decay == pytest.approx(0.99)
    assert config.ema.start_step == 10
    assert config.ema.update_every_n_steps == 2
    assert config.model.gradient_checkpointing is True


def test_parse_pcvr_train_config_uses_runtime_progress_log_interval_default(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(runtime=RuntimeExecutionConfig(progress_log_interval_steps=77))

    config = parse_pcvr_train_config([], config_type=PCVRTrainConfig, defaults=defaults)

    assert config.runtime.progress_log_interval_steps == 77


def test_build_train_model_configures_shared_flash_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        workflow_module,
        "configure_shared_flash_attention_runtime",
        lambda *, backend: captured.update({"backend": backend}),
    )
    monkeypatch.setattr(
        workflow_module.shared_modeling,
        "configure_rms_norm_runtime",
        lambda **kwargs: captured.update({"rms_norm": kwargs}),
    )

    class FakeModel:
        num_ns = 0

        def __init__(self, *, schema, config):
            del schema, config

        def to(self, device):
            del device
            return self

        def parameters(self):
            return []

    context = SimpleNamespace(
        model_class_name="FakeModel",
        model_type=FakeModel,
        data_dir=tmp_path / "data",
        ckpt_dir=tmp_path / "checkpoints",
        schema_path=tmp_path / "schema.json",
        device="cpu",
        reporter=None,
        config=PCVRTrainConfig(model=PCVRModelConfig(flash_attention_backend="tilelang")),
    )
    data_bundle = PCVRTrainDataBundle(
        train_loader="train",
        valid_loader="valid",
        dataset=SimpleNamespace(
            seq_domains=[],
            layout=SimpleNamespace(schema=object()),
        ),
    )

    model = build_train_model(context, data_bundle)

    assert captured == {
        "backend": "tilelang",
        "rms_norm": {"backend": "torch", "block_rows": 1},
    }
    assert isinstance(model, FakeModel)


def test_parse_pcvr_train_config_accepts_rms_norm_flags(tmp_path: Path) -> None:
    config = parse_pcvr_train_config(
        ["--model.rms_norm_backend", "triton", "--model.rms_norm_block_rows", "8"],
        config_type=PCVRTrainConfig,
        defaults=PCVRTrainConfig(),
    )

    assert config.model.rms_norm_backend == "triton"
    assert config.model.rms_norm_block_rows == 8


def test_parse_pcvr_train_config_accepts_flash_attention_backend_flag(tmp_path: Path) -> None:
    config = parse_pcvr_train_config(
        ["--model.flash_attention_backend", "tilelang"],
        config_type=PCVRTrainConfig,
        defaults=PCVRTrainConfig(),
    )

    assert config.model.flash_attention_backend == "tilelang"


def test_parse_pcvr_train_config_can_disable_default_rope(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(model=PCVRModelConfig(use_rope=True))

    config = parse_pcvr_train_config(
        ["--model.no_use_rope"],
        config_type=PCVRTrainConfig,
        defaults=defaults,
    )

    assert config.model.use_rope is False


def test_parse_pcvr_train_config_accepts_timestamp_auto_split(tmp_path: Path) -> None:
    config = parse_pcvr_train_config(
        [
            "--data.split_strategy",
            "timestamp_auto",
        ],
        config_type=PCVRTrainConfig,
        defaults=PCVRTrainConfig(),
    )

    assert config.data.split_strategy == "timestamp_auto"


def test_parse_pcvr_train_config_accepts_sampling_strategy(tmp_path: Path) -> None:
    config = parse_pcvr_train_config(
        ["--data.sampling_strategy", "row_group_sweep"],
        config_type=PCVRTrainConfig,
        defaults=PCVRTrainConfig(),
    )

    assert config.data.sampling_strategy == "row_group_sweep"


@pytest.mark.parametrize("flag", ["--optimizer.patience", "--optimizer.steps_per_epoch"])
def test_parse_pcvr_train_config_rejects_legacy_epoch_flags(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_config(
            [flag, "1"],
            config_type=PCVRTrainConfig,
            defaults=PCVRTrainConfig(),
        )


def test_parse_pcvr_train_config_uses_timestamp_auto_split_defaults(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(
        data=PCVRDataConfig(
            split_strategy="timestamp_auto",
        )
    )

    config = parse_pcvr_train_config([], config_type=PCVRTrainConfig, defaults=defaults)

    assert config.data.split_strategy == "timestamp_auto"


def test_parse_pcvr_train_config_rejects_symbiosis_ablation_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_config(
            [
                "--model.no_symbiosis_v2_use_dense_tokens",
                "--model.symbiosis_v2_recent_event_tokens",
                "16",
            ],
            config_type=PCVRTrainConfig,
            defaults=PCVRTrainConfig(),
        )


def test_symbiosis_package_parser_accepts_symbiosis_ablation_flags() -> None:
    symbiosis_module = load_module_from_path(REPO_ROOT / "experiments" / "symbiosis")

    config = parse_pcvr_train_config(
        [
            "--model.no_v2_use_dense_tokens",
            "--model.no_v2_use_missing_tokens",
            "--model.no_v2_use_sequence_stats_tokens",
            "--model.no_v2_use_metadata_attention_bias",
            "--model.no_v2_use_candidate_readout",
            "--model.v2_tokenization_mode",
            "group_compressed",
            "--model.v2_sparse_seed",
            "123",
            "--model.v2_recent_event_tokens",
            "16",
            "--model.v2_memory_event_tokens",
            "4",
            "--model.v3_memory_selection_mode",
            "stratified",
            "--model.v3_recent_event_tokens_by_domain",
            "seq_a:6,seq_b:8",
            "--model.v3_memory_event_tokens_by_domain",
            "seq_a:3,seq_b:4",
        ],
        config_type=symbiosis_module.SymbiosisTrainConfig,
        defaults=symbiosis_module.TRAIN_DEFAULTS,
    )

    model = config.model
    assert model.v2_use_dense_tokens is False
    assert model.v2_use_missing_tokens is False
    assert model.v2_use_sequence_stats_tokens is False
    assert model.v2_use_metadata_attention_bias is False
    assert model.v2_use_candidate_readout is False
    assert model.v2_tokenization_mode == "group_compressed"
    assert model.v2_sparse_seed == 123
    assert model.v2_recent_event_tokens == 16
    assert model.v2_memory_event_tokens == 4
    assert model.v3_enabled is True
    assert model.v3_memory_selection_mode == "stratified"
    assert model.v3_recent_event_tokens_by_domain == "seq_a:6,seq_b:8"
    assert model.v3_memory_event_tokens_by_domain == "seq_a:3,seq_b:4"


@pytest.mark.parametrize("dense_optimizer_type", ["orthogonal_adamw", "fused_adamw", "muon"])
def test_parse_pcvr_train_config_accepts_loss_terms_and_optimizer_flags(tmp_path: Path, dense_optimizer_type: str) -> None:
    config = parse_pcvr_train_config(
        [
            "--loss.terms.0.name",
            "bce",
            "--loss.terms.0.kind",
            "bce",
            "--loss.terms.0.weight",
            "1.0",
            "--data.eval_every_n_steps",
            "5000",
            "--optimizer.dense_optimizer_type",
            dense_optimizer_type,
            "--optimizer.patience_steps",
            "77",
            "--optimizer.scheduler_type",
            "cosine",
            "--optimizer.warmup_steps",
            "256",
            "--optimizer.min_lr_ratio",
            "0.1",
        ],
        config_type=PCVRTrainConfig,
        defaults=PCVRTrainConfig(),
    )

    assert config.loss.terms[0].name == "bce"
    assert config.loss.terms[0].kind == "bce"
    assert config.loss.terms[0].weight == pytest.approx(1.0)
    assert config.data.eval_every_n_steps == 5000
    assert config.optimizer.dense_optimizer_type == dense_optimizer_type
    assert config.optimizer.patience_steps == 77
    assert config.optimizer.scheduler_type == "cosine"
    assert config.optimizer.warmup_steps == 256
    assert config.optimizer.min_lr_ratio == pytest.approx(0.1)


def test_parse_pcvr_train_config_uses_scheduler_defaults(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(
        optimizer=PCVROptimizerConfig(
            scheduler_type="linear",
            warmup_steps=128,
            min_lr_ratio=0.25,
        )
    )

    config = parse_pcvr_train_config([], config_type=PCVRTrainConfig, defaults=defaults)

    assert config.optimizer.scheduler_type == "linear"
    assert config.optimizer.warmup_steps == 128
    assert config.optimizer.min_lr_ratio == pytest.approx(0.25)


def test_parse_pcvr_train_config_rejects_data_pipeline_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_config(
            ["--data_pipeline.transforms", "sequence_crop,feature_mask"],
            config_type=PCVRTrainConfig,
            defaults=PCVRTrainConfig(),
        )


def test_parse_pcvr_train_config_uses_typed_data_pipeline_defaults(
    tmp_path: Path,
) -> None:
    defaults = PCVRTrainConfig(
        data_pipeline=PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode="lru", max_batches=32),
            seed=77,
            transforms=(
                PCVRSequenceCropConfig(
                    views_per_row=2,
                    seq_window_mode="random_tail",
                    seq_window_min_len=8,
                ),
                PCVRFeatureMaskConfig(probability=0.05),
            ),
        )
    )

    config = parse_pcvr_train_config([], config_type=PCVRTrainConfig, defaults=defaults)

    assert config.data_pipeline.cache.mode == "lru"
    assert config.data_pipeline.seed == 77
    assert config.data_pipeline.transforms[0].name == "sequence_crop"


def test_pcvr_train_config_serializes_structured_data_pipeline() -> None:
    defaults = PCVRTrainConfig(
        data_pipeline=PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode="lru", max_batches=32),
            seed=77,
            strict_time_filter=False,
            transforms=(
                PCVRSequenceCropConfig(
                    views_per_row=2,
                    seq_window_mode="random_tail",
                    seq_window_min_len=8,
                ),
                PCVRFeatureMaskConfig(probability=0.05),
                PCVRNonSequentialSparseDropoutConfig(probability=0.15),
                PCVRDomainDropoutConfig(probability=0.1),
            ),
        )
    )

    flat_config = defaults.model_dump(mode="json")

    assert "data_pipeline" in flat_config
    assert flat_config["data_pipeline"]["transforms"][0]["name"] == "sequence_crop"
    assert flat_config["data_pipeline"]["transforms"][1]["name"] == "feature_mask"
    assert flat_config["data_pipeline"]["transforms"][2]["name"] == "nonseq_sparse_dropout"
    assert flat_config["data_pipeline"]["transforms"][3]["name"] == "domain_dropout"


def test_pcvr_train_config_serializes_optimizer_schedule_fields() -> None:
    flat_config = PCVRTrainConfig(
        optimizer=PCVROptimizerConfig(
            patience_steps=512,
            scheduler_type="cosine",
            warmup_steps=64,
            min_lr_ratio=0.2,
        )
    ).model_dump(mode="json")

    assert flat_config["optimizer"]["patience_steps"] == 512
    assert flat_config["optimizer"]["scheduler_type"] == "cosine"
    assert flat_config["optimizer"]["warmup_steps"] == 64
    assert flat_config["optimizer"]["min_lr_ratio"] == pytest.approx(0.2)


def test_pcvr_train_config_serializes_ema_fields() -> None:
    flat_config = PCVRTrainConfig(
        ema=PCVREMAConfig(
            enabled=True,
            decay=0.995,
            start_step=128,
            update_every_n_steps=4,
        )
    ).model_dump(mode="json")

    assert flat_config["ema"]["enabled"] is True
    assert flat_config["ema"]["decay"] == pytest.approx(0.995)
    assert flat_config["ema"]["start_step"] == 128
    assert flat_config["ema"]["update_every_n_steps"] == 4


def test_pcvr_ema_config_validates_values() -> None:
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\)"):
        PCVREMAConfig(decay=1.0)
    with pytest.raises(ValueError, match="start_step"):
        PCVREMAConfig(start_step=-1)
    with pytest.raises(ValueError, match="update_every_n_steps"):
        PCVREMAConfig(update_every_n_steps=0)


def test_pcvr_train_config_serializes_runtime_determinism_field() -> None:
    flat_config = PCVRTrainConfig(runtime=RuntimeExecutionConfig(deterministic=False)).model_dump(mode="json")

    assert flat_config["runtime"]["deterministic"] is False


def test_pcvr_train_config_serializes_data_split_fields() -> None:
    flat_config = PCVRTrainConfig(
        data=PCVRDataConfig(
            train_steps_per_sweep=128,
            split_strategy="timestamp_auto",
        )
    ).model_dump(mode="json")

    assert flat_config["data"]["eval_every_n_steps"] == 5000
    assert flat_config["data"]["train_steps_per_sweep"] == 128
    assert flat_config["data"]["split_strategy"] == "timestamp_auto"


def test_pcvr_train_config_serializes_gradient_checkpointing_field() -> None:
    flat_config = PCVRTrainConfig(
        model=PCVRModelConfig(gradient_checkpointing=True)
    ).model_dump(mode="json")

    assert flat_config["model"]["gradient_checkpointing"] is True


def test_pcvr_train_config_serializes_rms_norm_fields() -> None:
    flat_config = PCVRTrainConfig(
        model=PCVRModelConfig(rms_norm_backend="triton", rms_norm_block_rows=8)
    ).model_dump(mode="json")

    assert flat_config["model"]["rms_norm_backend"] == "triton"
    assert flat_config["model"]["rms_norm_block_rows"] == 8


def test_pcvr_train_config_serializes_flash_attention_backend_field() -> None:
    flat_config = PCVRTrainConfig(
        model=PCVRModelConfig(flash_attention_backend="tilelang")
    ).model_dump(mode="json")

    assert flat_config["model"]["flash_attention_backend"] == "tilelang"


def test_pcvr_train_config_serializes_loss_terms() -> None:
    flat_config = PCVRTrainConfig(
        loss=PCVRLossConfig(
            terms=(
                PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),
                PCVRLossTermConfig(name="aux", kind="model", weight=0.05),
            )
        )
    ).model_dump(mode="json")

    assert flat_config["loss"]["terms"] == [
        {
            "name": "bce",
            "kind": "bce",
            "weight": 1.0,
            "focal_alpha": 0.1,
            "focal_gamma": 2.0,
            "temperature": 1.0,
        },
        {
            "name": "aux",
            "kind": "model",
            "weight": 0.05,
            "focal_alpha": 0.1,
            "focal_gamma": 2.0,
            "temperature": 1.0,
        },
    ]
