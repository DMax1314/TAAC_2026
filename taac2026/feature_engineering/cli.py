from __future__ import annotations

import argparse
from pathlib import Path

from ..utils import ensure_dir, write_json
from .schema import build_schema_artifacts, print_human_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 TAAC 2026 parquet 数据集，并导出 schema 摘要与特征字典。")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/datasets--TAAC2026--data_sample_1000/snapshots/2f0ddba721a8323495e73d5229c836df5d603b39/sample_data.parquet",
        help="Parquet 数据集路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_engineering",
        help="导出目录，默认同时写出 schema_summary.json 与 feature_dictionary.json。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_schema_artifacts(args.dataset_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    schema_path = output_dir / "schema_summary.json"
    dictionary_path = output_dir / "feature_dictionary.json"
    write_json(schema_path, artifacts["schema_summary"])
    write_json(dictionary_path, artifacts["feature_dictionary"])

    print_human_summary(artifacts["schema_summary"])
    print(f"schema_summary_written_to={schema_path}")
    print(f"feature_dictionary_written_to={dictionary_path}")


if __name__ == "__main__":
    main()