from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from taac2026.application.packaging.cli import build_training_bundle
from taac2026.infrastructure.checkpoints import (
    build_checkpoint_dir_name,
    checkpoint_step,
    resolve_checkpoint_path,
    validate_checkpoint_dir_name,
    write_checkpoint_sidecars,
)
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.domain.config import PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.domain.sidecar import (
    PCVR_TRAIN_CONFIG_FORMAT,
    build_pcvr_train_config_sidecar,
)
from taac2026.infrastructure.modeling.model_contract import (
    build_feature_specs,
    build_pcvr_model_specs,
    dataset_schema_path,
    load_ns_groups,
    parse_seq_max_lens,
    resolve_schema_path,
)
from taac2026.application.evaluation.runtime import load_train_config
from taac2026.infrastructure.io.json import dumps, loads
from tests.support.experiment_matrix import ExperimentCase, REPO_ROOT, discover_pcvr_experiment_cases, load_model_module


EXPERIMENT_CASES = discover_pcvr_experiment_cases()


def _schema(entries: list[tuple[int, int, int]]) -> SimpleNamespace:
    return SimpleNamespace(entries=entries)


def _make_schema(user_fids: list[int], item_fids: list[int]) -> PCVRSchema:
    return PCVRSchema(
        format="raw_parquet",
        user_int=tuple([fid, 10, 1] for fid in user_fids),
        item_int=tuple([fid, 10, 1] for fid in item_fids),
        user_dense=[[4, 4]],
        seq={},
    )

def _code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as archive:
        return set(archive.namelist())


def _code_package_manifest(code_package_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as archive:
        return loads(archive.read("project/.taac_training_manifest.json"))


@pytest.fixture(scope="module", params=EXPERIMENT_CASES, ids=lambda case: case.path)
def experiment_case(request) -> ExperimentCase:
    return request.param


@pytest.fixture(scope="module")
def loaded_experiment(experiment_case: ExperimentCase):
    return load_experiment_package(experiment_case.path)


@pytest.fixture(scope="module")
def loaded_model_module(experiment_case: ExperimentCase):
    return load_model_module(experiment_case)


@pytest.fixture(scope="module")
def built_bundle(experiment_case: ExperimentCase, tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp(f"bundle_{Path(experiment_case.path).name}")
    result = build_training_bundle(experiment_case.path, output_dir=output_dir, root=REPO_ROOT)
    return result, _code_package_manifest(result.code_package_path), _code_package_names(result.code_package_path)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("", {}),
        (" , ", {}),
        ("seq_a:4", {"seq_a": 4}),
        ("seq_a:4,seq_b:8", {"seq_a": 4, "seq_b": 8}),
        (" seq_a : 4 ", {"seq_a": 4}),
        ("seq_a:4,,seq_b:8,", {"seq_a": 4, "seq_b": 8}),
        ("seq_a:004", {"seq_a": 4}),
        ("seq_a:4, seq_a:8", {"seq_a": 8}),
        ("seq_a:0", {"seq_a": 0}),
        ("seq_a:3,\nseq_b:5", {"seq_a": 3, "seq_b": 5}),
    ],
)
def test_parse_seq_max_lens_cases(raw_value: str, expected: dict[str, int]) -> None:
    assert parse_seq_max_lens(raw_value) == expected


@pytest.mark.parametrize(
    ("entries", "vocab_sizes", "expected"),
    [
        ([(10, 0, 1)], [11], [(11, 0, 1)]),
        ([(10, 0, 2)], [11, 21], [(21, 0, 2)]),
        ([(10, 0, 2), (20, 2, 1)], [11, 21, 7], [(21, 0, 2), (7, 2, 1)]),
        ([(10, 1, 2)], [3, 5, 8], [(8, 1, 2)]),
        ([(10, 0, 3)], [4, 4, 4], [(4, 0, 3)]),
        ([(10, 2, 2)], [1, 1, 9, 8], [(9, 2, 2)]),
        ([(10, 0, 1), (20, 1, 2)], [2, 6, 5], [(2, 0, 1), (6, 1, 2)]),
        ([(10, 3, 1)], [1, 2, 3, 10], [(10, 3, 1)]),
    ],
)
def test_build_feature_specs_cases(
    entries: list[tuple[int, int, int]],
    vocab_sizes: list[int],
    expected: list[tuple[int, int, int]],
) -> None:
    assert build_feature_specs(_schema(entries), vocab_sizes) == expected


@pytest.mark.parametrize(
    "scenario",
    [
        "explicit",
        "explicit_missing",
        "fallback",
        "fallback_missing",
    ],
)
def test_resolve_schema_path_cases(tmp_path: Path, scenario: str) -> None:
    dataset_dir = tmp_path / "data_dir"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "train.parquet"
    dataset_file.write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints" / "global_step1"
    checkpoint_dir.mkdir(parents=True)
    explicit_path = tmp_path / "explicit_schema.json"

    if scenario == "explicit":
        explicit_path.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(explicit_path, fallback=dataset_dir / "schema.json")
        assert actual == explicit_path.resolve()
        return

    if scenario == "explicit_missing":
        with pytest.raises(FileNotFoundError, match=r"explicit path"):
            resolve_schema_path(explicit_path, fallback=dataset_dir / "schema.json")
        return

    checkpoint_schema = checkpoint_dir / "schema.json"
    if scenario == "fallback":
        checkpoint_schema.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(None, fallback=checkpoint_schema)
        assert actual == checkpoint_schema.resolve()
        return

    with pytest.raises(FileNotFoundError, match=r"fallback path"):
        resolve_schema_path(None, fallback=checkpoint_schema)


def test_dataset_schema_path_uses_dataset_dir_or_parent(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data_dir"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "train.parquet"
    dataset_file.write_text("", encoding="utf-8")

    assert dataset_schema_path(dataset_dir) == (dataset_dir / "schema.json").resolve()
    assert dataset_schema_path(dataset_file) == (dataset_dir / "schema.json").resolve()


@pytest.mark.parametrize(
    ("user_count", "item_count"),
    [(1, 1), (2, 3), (4, 2)],
)
def test_load_ns_groups_defaults_to_singletons_when_disabled(user_count: int, item_count: int) -> None:
    schema = _make_schema(
        user_fids=[100 + index for index in range(user_count)],
        item_fids=[200 + index for index in range(item_count)],
    )

    user_groups, item_groups = load_ns_groups(
        schema,
        PCVRNSConfig(grouping_strategy="singleton", user_groups={}, item_groups={}),
    )

    assert user_groups == [[index] for index in range(user_count)]
    assert item_groups == [[index] for index in range(item_count)]


@pytest.mark.parametrize(
    ("payload", "expected_user", "expected_item"),
    [
        (
            {"user_groups": {"u": [20, 10]}, "item_groups": {"i": [7]}},
            [[1, 0]],
            [[0]],
        ),
        (
            {"user_groups": {"u1": [30], "u2": [20, 10]}, "item_groups": {"i1": [8], "i2": [7]}},
            [[2], [1, 0]],
            [[1], [0]],
        ),
    ],
)
def test_load_ns_groups_maps_feature_ids_preserves_declared_order(
    payload: dict[str, dict[str, list[int]]],
    expected_user: list[list[int]],
    expected_item: list[list[int]],
) -> None:
    schema = _make_schema(user_fids=[10, 20, 30], item_fids=[7, 8])

    user_groups, item_groups = load_ns_groups(
        schema,
        PCVRNSConfig(grouping_strategy="explicit", **payload),
    )

    assert user_groups == expected_user
    assert item_groups == expected_item


@pytest.mark.parametrize(
    ("payload", "missing_name"),
    [
        ({"user_groups": {"u": [999]}, "item_groups": {"i": [7]}}, "999"),
        ({"user_groups": {"u": [10]}, "item_groups": {"i": [999]}}, "999"),
    ],
)
def test_load_ns_groups_raises_for_unknown_feature_ids(
    payload: dict[str, dict[str, list[int]]],
    missing_name: str,
) -> None:
    schema = _make_schema(user_fids=[10], item_fids=[7])

    with pytest.raises(KeyError, match=missing_name):
        load_ns_groups(
            schema,
            PCVRNSConfig(grouping_strategy="explicit", **payload),
        )


def test_build_pcvr_model_specs_compiles_schema_derived_inputs() -> None:
    schema = _make_schema(user_fids=[10, 20], item_fids=[7])
    specs = build_pcvr_model_specs(
        schema,
        PCVRNSConfig(
            grouping_strategy="explicit",
            user_groups={"u": [20, 10]},
            item_groups={"i": [7]},
        ),
    )

    assert specs.user_int_feature_specs == [(10, 0, 1), (10, 1, 1)]
    assert specs.item_int_feature_specs == [(10, 0, 1)]
    assert specs.user_dense_dim == 4
    assert specs.item_dense_dim == 0
    assert specs.seq_vocab_sizes == {}
    assert specs.user_ns_groups == [[1, 0]]
    assert specs.item_ns_groups == [[0]]


@pytest.mark.parametrize(
    ("path_value", "expected"),
    [
        (Path("global_step0"), 0),
        (Path("global_step12.layer=2"), 12),
        (Path("global_step3.AUC=0.95") / "model.safetensors", 3),
        (Path("global_step0007"), 7),
        (Path("global_step9.hidden=64.extra_token"), 9),
        (Path("best_model"), -1),
        (Path("invalid_parent") / "model.safetensors", -1),
    ],
)
def test_checkpoint_step_cases(path_value: Path, expected: int) -> None:
    assert checkpoint_step(path_value) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        "best",
        "globalstep1",
        "global_step",
        "global_step1 space",
        "global_step1!",
        "global_step1/child",
        "g" * 301,
    ],
)
def test_validate_checkpoint_dir_name_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_checkpoint_dir_name(name)


@pytest.mark.parametrize(
    ("global_step", "params", "expected"),
    [
        (0, None, "global_step0"),
        (1, {"layer": 2}, "global_step1"),
        (1, {"auc": 0.9}, "global_step1.AUC=0.9"),
        (7, {"AUC": "0.912340", "unused": 9}, "global_step7.AUC=0.91234"),
        (9, {"layer": "02", "head": "04"}, "global_step9"),
        (12, {"layer": 2, "head": 4, "hidden": 64, "auc": 0.9501234}, "global_step12.AUC=0.950123"),
    ],
)
def test_build_checkpoint_dir_name_cases(
    global_step: int,
    params: dict[str, object] | None,
    expected: str,
) -> None:
    assert build_checkpoint_dir_name(global_step, params) == expected


def test_build_checkpoint_dir_name_rejects_negative_global_step() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_checkpoint_dir_name(-1)


@pytest.mark.parametrize(
    "scenario",
    [
        "explicit_file",
        "explicit_dir",
        "latest_step",
        "direct_checkpoint",
        "missing",
    ],
)
def test_resolve_checkpoint_path_cases(tmp_path: Path, scenario: str) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    if scenario == "explicit_file":
        candidate = tmp_path / "manual.safetensors"
        candidate.write_text("manual", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir, candidate) == candidate.resolve()
        return

    if scenario == "explicit_dir":
        candidate_dir = tmp_path / "manual_dir"
        candidate_dir.mkdir()
        model_path = candidate_dir / "model.safetensors"
        model_path.write_text("manual", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir, candidate_dir) == model_path.resolve()
        return

    if scenario == "latest_step":
        older = run_dir / "global_step1.AUC=0.9"
        newer = run_dir / "global_step3.AUC=0.95"
        older.mkdir()
        newer.mkdir()
        (older / "model.safetensors").write_text("old", encoding="utf-8")
        (newer / "model.safetensors").write_text("new", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir) == (newer / "model.safetensors").resolve()
        return

    if scenario == "direct_checkpoint":
        direct_model = run_dir / "model.safetensors"
        direct_model.write_text("direct", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir) == direct_model.resolve()
        return

    with pytest.raises(FileNotFoundError, match=r"no model\.safetensors checkpoint"):
        resolve_checkpoint_path(run_dir)


def test_write_checkpoint_sidecars_writes_both_sidecars(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step1"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"schema": true}\n', encoding="utf-8")

    written = write_checkpoint_sidecars(
        checkpoint_dir,
        schema_path=schema_path,
        train_config=PCVRTrainConfig(
            model=PCVRModelConfig(
                d_model=64,
                ns=PCVRNSConfig(
                    grouping_strategy="explicit",
                    user_groups={"u": [10, 20]},
                    item_groups={"i": [7]},
                ),
            ),
        ),
    )

    assert set(written) == {"schema", "train_config"}
    assert (checkpoint_dir / "schema.json").exists()
    payload = loads((checkpoint_dir / "train_config.json").read_bytes())
    assert payload["train_config_format"] == PCVR_TRAIN_CONFIG_FORMAT
    assert payload["train_config"]["model"]["ns"]["grouping_strategy"] == "explicit"
    assert payload["train_config"]["model"]["ns"]["user_groups"] == {"u": [10, 20]}
    assert payload["train_config"]["model"]["ns"]["item_groups"] == {"i": [7]}


def test_write_checkpoint_sidecars_requires_existing_schema(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step1"
    missing_schema = tmp_path / "missing_schema.json"

    with pytest.raises(FileNotFoundError, match="schema"):
        write_checkpoint_sidecars(
            checkpoint_dir,
            schema_path=missing_schema,
            train_config=PCVRTrainConfig(),
        )


def test_build_pcvr_train_config_sidecar_adds_framework_metadata() -> None:
    structured_config = PCVRTrainConfig()

    payload = build_pcvr_train_config_sidecar(structured_config)

    assert payload["train_config_format"] == PCVR_TRAIN_CONFIG_FORMAT
    assert payload["framework_name"] == "taac2026"
    assert "framework_version" not in payload
    assert payload["train_config"]["model"]["d_model"] == structured_config.model.d_model
    assert payload["train_config"]["model"]["ns"] == structured_config.model.ns.model_dump(mode="json")


def test_load_train_config_requires_current_payload(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step1"
    checkpoint_dir.mkdir()
    structured_config = PCVRTrainConfig()
    (checkpoint_dir / "train_config.json").write_text(dumps(build_pcvr_train_config_sidecar(structured_config)), encoding="utf-8")

    loaded = load_train_config(PCVRTrainConfig, checkpoint_dir)

    assert loaded.model.d_model == structured_config.model.d_model
    assert loaded.model.ns.grouping_strategy == structured_config.model.ns.grouping_strategy


def test_load_train_config_rejects_flat_payload(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step1"
    checkpoint_dir.mkdir()
    flat_payload = {
        "train_config_format": PCVR_TRAIN_CONFIG_FORMAT,
        "framework_name": "taac2026",
        "framework_version": "test",
        "train_config": {"d_model": 64, "ns_grouping_strategy": "explicit"},
    }
    (checkpoint_dir / "train_config.json").write_text(dumps(flat_payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_train_config(PCVRTrainConfig, checkpoint_dir)


@pytest.mark.parametrize("identifier_kind", ["path", "path_object"])
def test_load_experiment_package_accepts_path_and_module_identifiers(
    experiment_case: ExperimentCase,
    identifier_kind: str,
) -> None:
    identifier = experiment_case.path if identifier_kind == "path" else (REPO_ROOT / experiment_case.path)
    experiment = load_experiment_package(identifier)

    assert experiment.name == experiment_case.name
    assert experiment.package_dir == (REPO_ROOT / experiment_case.path).resolve()


def test_experiment_package_contracts(loaded_experiment, experiment_case: ExperimentCase) -> None:
    train_defaults = loaded_experiment.train_defaults.model_dump(mode="json")

    assert loaded_experiment.name == experiment_case.name
    assert loaded_experiment.package_dir == (REPO_ROOT / experiment_case.path).resolve()
    assert loaded_experiment.train_defaults is not None
    assert loaded_experiment.kind == "pcvr"
    assert loaded_experiment.model_class_name == experiment_case.model_class
    assert train_defaults["model"]["ns"]["grouping_strategy"] == "explicit"
    assert train_defaults["model"]["ns"]["user_groups"]
    assert train_defaults["model"]["ns"]["item_groups"]
    assert "num_hyformer_blocks" not in train_defaults


def test_model_module_contracts(loaded_model_module, experiment_case: ExperimentCase) -> None:
    assert hasattr(loaded_model_module, experiment_case.model_class)
    if experiment_case.path != "experiments/baseline":
        assert not hasattr(loaded_model_module, "PCVRHyFormer")


def test_ns_group_config_has_required_keys(experiment_case: ExperimentCase) -> None:
    experiment = load_experiment_package(experiment_case.path)
    ns = experiment.train_defaults.model.ns

    assert ns.grouping_strategy == "explicit"
    assert isinstance(ns.user_groups, dict)
    assert isinstance(ns.item_groups, dict)
    assert all(isinstance(group, list) for group in ns.user_groups.values())
    assert all(isinstance(group, list) for group in ns.item_groups.values())
    assert all(all(isinstance(feature_id, int) for feature_id in group) for group in ns.user_groups.values())
    assert all(all(isinstance(feature_id, int) for feature_id in group) for group in ns.item_groups.values())


def test_bundle_manifest_points_to_selected_experiment(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    result, manifest, _names = built_bundle

    assert result.output_dir.is_dir()
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    assert manifest["bundle_format"] == "taac2026-training"
    assert manifest["bundled_experiment_path"] == experiment_case.path
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["code_package"] == "code_package.zip"


def test_bundle_contains_model_and_explicit_ns_config(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    _result, _manifest, names = built_bundle

    assert f"project/{experiment_case.path}/__init__.py" in names
    assert f"project/{experiment_case.path}/model.py" in names
    assert f"project/{experiment_case.path}/ns_groups.json" not in names
    assert "project/src/taac2026/application/training/cli.py" in names
    assert "project/src/taac2026/application/training/args.py" in names
    assert "project/src/taac2026/application/training/workflow.py" in names


def test_bundle_excludes_package_local_runtime_wrappers(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    _result, _manifest, names = built_bundle

    assert "project/run.sh" not in names
    assert f"project/{experiment_case.path}/run.sh" not in names
    assert f"project/{experiment_case.path}/train.py" not in names
    assert f"project/{experiment_case.path}/trainer.py" not in names
