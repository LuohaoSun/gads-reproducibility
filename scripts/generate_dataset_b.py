"""Generate semi-synthetic Dataset B bundles from UCI SECOM."""

from __future__ import annotations

import argparse
from pathlib import Path

from shap_diff_analysis.secom import (
    DEFAULT_RAW_DIR,
    SCENARIO_NAMES,
    generate_dataset_b,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Dataset B generation."""
    parser = argparse.ArgumentParser(description="Generate Dataset B (SECOM semi-synthetic) bundles.")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_NAMES),
        required=True,
        help="Fault-injection scenario to generate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for feature roles and label generation.")
    parser.add_argument(
        "--group-seed",
        type=int,
        default=2024,
        help="Independent seed for pseudo-EQP partitioning.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing downloaded SECOM raw files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/dataset_b"),
        help="Directory for generated parquet files and manifests.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for Dataset B generation."""
    args = parse_args()
    scenario = args.scenario
    bundle = generate_dataset_b(
        scenario=scenario,
        seed=args.seed,
        group_seed=args.group_seed,
        raw_dir=args.raw_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dataset_b_{scenario}_seed{args.seed}_group{args.group_seed}"
    data_path = args.output_dir / f"{stem}.parquet"
    manifest_path = args.output_dir / f"{stem}.manifest.json"

    bundle.data.to_parquet(data_path, index=False)
    write_manifest(bundle, manifest_path)

    print(f"Scenario: {scenario}")
    print(f"Rows: {len(bundle.data)} | Features: {len(bundle.process_features)}")
    print(f"Abnormal devices: {bundle.abnormal_devices}")
    print(f"Wrote data: {data_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
