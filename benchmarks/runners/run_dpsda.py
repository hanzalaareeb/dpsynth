#!/usr/bin/env python3
"""Run the DPSDA Tab-PE Adult benchmark."""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from pe.api import TabularAPI
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
from pe.embedding import TabularEmbedding
from pe.histogram import NearestNeighbors
from pe.population import CompositePopulation, PEPopulation
from pe.runner import PE
from pe.data import TabularCSV, TabularColumnType

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(value: str) -> Path:
    """Resolve a path from the benchmark root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the benchmark YAML configuration."""
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must be a YAML mapping.")
    return config


def read_csv_header(csv_path: Path) -> list[str]:
    """Read only the header row of a CSV file."""
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        return next(csv.reader(csv_file))


def set_seed(seed: int) -> None:
    """Seed random-number generators used by DPSDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_public_tabular_info(
    config: dict[str, Any],
    private_data: TabularCSV,
) -> dict[str, dict[str, Any]]:
    """Build base DPSDA feature schema from domain + tabular metadata."""
    dataset = config["dataset"]
    dpsda_config = config["dpsda"]

    domain_path = repo_path(dataset["domain_path"])
    with domain_path.open(encoding="utf-8") as domain_file:
        domain = yaml.safe_load(domain_file)

    feature_columns = list(private_data.metadata.feature_columns)
    categorical_columns = set(private_data.metadata.cat_columns)
    numeric_columns = set(dataset["numeric_columns"])

    internal_numeric_type = dpsda_config["internal_numeric_type"]
    if internal_numeric_type != "float":
        raise ValueError(
            "This adapter currently expects "
            "dpsda.internal_numeric_type: float."
        )

    info: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        column_domain = domain[column]
        if column in categorical_columns:
            info[column] = {
                "categories": list(column_domain["possible_values"]),
                "type": TabularColumnType.CATEGORICAL,
            }
        elif column in numeric_columns:
            info[column] = {
                "min": float(column_domain["min_value"]),
                "max": float(column_domain["max_value"]),
                "type": TabularColumnType.FLOAT,
            }
        else:
            raise ValueError(
                f"Feature column {column!r} is neither categorical nor numerical."
            )

    return info


def prepare_dpsda_inputs(
    config: dict[str, Any],
    seed: int,
) -> tuple[TabularCSV, dict[str, dict[str, Any]], int]:
    """Load private data and build DPSDA-compatible feature metadata."""
    dataset = config["dataset"]

    set_seed(seed)

    private_data = TabularCSV(
        csv_path=str(repo_path(dataset["train_path"])),
        metadata_path=str(repo_path(dataset["metadata_path"])),
    )

    expected_private_rows = config["privacy"]["delta"]["n"]
    actual_private_rows = len(private_data.data_frame)
    if actual_private_rows != expected_private_rows:
        raise ValueError(
            f"Expected {expected_private_rows} private rows, "
            f"found {actual_private_rows}."
        )

    label_column = dataset["label_column"]
    actual_label_order = [
        label_info.column_values[label_column]
        for label_info in private_data.metadata.label_info
    ]
    expected_label_order = config[
        "public_side_information"
    ]["label_distribution"]["label_order"]
    if actual_label_order != expected_label_order:
        raise ValueError(
            "DPSDA label ordering does not match configured public "
            "label distribution.\n"
            f"Expected: {expected_label_order}\n"
            f"Actual:   {actual_label_order}"
        )

    public_info = build_public_tabular_info(config, private_data)

    expected_feature_columns = (
        set(dataset["categorical_columns"])
        | set(dataset["numeric_columns"])
    )
    if set(public_info) != expected_feature_columns:
        raise ValueError(
            "DPSDA public feature information does not match "
            "configured feature columns."
        )

    return private_data, public_info, seed


def build_dpsda_runner(
    config: dict[str, Any],
    private_data: TabularCSV,
    public_info: dict[str, dict[str, Any]],
) -> PE:
    """Construct the upstream Adult Tab-PE architecture."""
    dpsda_config = config["dpsda"]
    histogram_config = dpsda_config["histogram"]

    num_iterations = int(dpsda_config["num_iterations"])
    schedule_length = int(dpsda_config["num_samples_schedule_length"])
    sampling_iterations = int(dpsda_config["sampling_iterations"])
    variation_degree = int(dpsda_config["variation_degree"])

    if num_iterations != schedule_length:
        raise ValueError(
            "dpsda.num_iterations must equal "
            "dpsda.num_samples_schedule_length."
        )
    if not 0 < sampling_iterations <= num_iterations:
        raise ValueError(
            "dpsda.sampling_iterations must be between 1 and "
            "dpsda.num_iterations."
        )
    if variation_degree <= 0:
        raise ValueError("dpsda.variation_degree must be positive.")
    if histogram_config["type"] != "nearest_neighbors":
        raise ValueError("Only the nearest-neighbors histogram is supported.")

    api = TabularAPI(
        info=public_info,
        mutation_rate_init=float(dpsda_config["mutation_rate_initial"]),
        mutation_rate_final=float(dpsda_config["mutation_rate_final"]),
        decay_type=dpsda_config["mutation_decay_type"],
        gamma=float(dpsda_config["gamma"]),
        num_iterations=num_iterations,
    )

    embedding = TabularEmbedding(info=public_info)

    histogram = NearestNeighbors(
        embedding=embedding,
        mode=histogram_config["mode"],
        lookahead_degree=int(histogram_config["lookahead_degree"]),
        backend=histogram_config["backend"],
    )

    sampling_population = PEPopulation(
        api=api,
        initial_variation_api_fold=0,
        next_variation_api_fold=1,
        keep_selected=False,
        selection_mode="sample",
        histogram_threshold=0,
    )

    ranking_population = PEPopulation(
        api=api,
        initial_variation_api_fold=variation_degree,
        next_variation_api_fold=variation_degree,
        keep_selected=True,
        selection_mode="rank",
    )

    populations = (
        [sampling_population] * sampling_iterations
        + [ranking_population] * (num_iterations - sampling_iterations)
    )
    if len(populations) != schedule_length:
        raise ValueError(
            f"Expected {schedule_length} population entries, "
            f"constructed {len(populations)}."
        )

    population = CompositePopulation(populations=populations)
    return PE(
        priv_data=private_data,
        population=population,
        histogram=histogram,
        callbacks=[],
        loggers=[],
    )


def make_smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    """Create a reduced config for a smoke test."""
    smoke_config = copy.deepcopy(config)
    smoke_config["experiment"]["synthetic_rows"] = 100
    smoke_config["dpsda"]["num_iterations"] = 2
    smoke_config["dpsda"]["num_samples_schedule_length"] = 2
    smoke_config["dpsda"]["private_histogram_rounds"] = 1
    # both entries use sampling in smoke mode
    smoke_config["dpsda"]["sampling_iterations"] = 2
    return smoke_config


def validate_config(config: dict[str, Any]) -> None:
    """Validate benchmark files and experiment parameters."""
    dataset = config["dataset"]
    privacy = config["privacy"]
    experiment = config["experiment"]

    required_paths = {
        "manifest": repo_path(dataset["manifest"]),
        "train": repo_path(dataset["train_path"]),
        "test": repo_path(dataset["test_path"]),
        "metadata": repo_path(dataset["metadata_path"]),
        "domain": repo_path(dataset["domain_path"]),
        "DPSDA checkout": REPO_ROOT / "benchmarks/external/DPSDA",
    }
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    columns = dataset["columns"]
    train_header = read_csv_header(required_paths["train"])
    test_header = read_csv_header(required_paths["test"])

    if train_header != columns:
        raise ValueError(
            "Adult training columns do not match dataset.columns.\n"
            f"Expected: {columns}\n"
            f"Actual: {train_header}"
        )
    if test_header != columns:
        raise ValueError(
            "Adult test columns do not match dataset.columns.\n"
            f"Expected: {columns}\n"
            f"Actual:   {test_header}"
        )

    with required_paths["domain"].open(encoding="utf-8") as domain_file:
        domain = yaml.safe_load(domain_file)
    if set(domain) != set(columns):
        raise ValueError(
            "Domain columns do not match dataset.columns.\n"
            f"Missing from domain: {sorted(set(columns) - set(domain))}\n"
            f"Extra in domain: {sorted(set(domain) - set(columns))}"
        )

    with required_paths["metadata"].open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    metadata_columns = (
        metadata["cat_columns"]
        + metadata["int_columns"]
        + metadata["float_columns"]
        + metadata["label_columns"]
    )
    if set(metadata_columns) != set(columns):
        raise ValueError("DPSDA metadata columns do not match dataset.columns.")

    synthetic_rows = experiment["synthetic_rows"]
    if not isinstance(synthetic_rows, int) or synthetic_rows <= 0:
        raise ValueError("experiment.synthetic_rows must be a positive integer.")

    epsilon = privacy["epsilon"]
    delta = privacy["delta"]["value"]
    n = privacy["delta"]["n"]
    expected_delta = 1.0 / (n * math.log(n))

    if epsilon <= 0:
        raise ValueError("privacy.epsilon must be positive.")
    if not math.isclose(delta, expected_delta, rel_tol=1e-12):
        raise ValueError(
            f"Configured delta {delta} does not match 1/(n log n) "
            f"for n={n}: {expected_delta}"
        )

    dpsda = config["dpsda"]
    schedule_length = dpsda["num_samples_schedule_length"]
    private_rounds = dpsda["private_histogram_rounds"]
    if private_rounds != schedule_length - 1:
        raise ValueError(
            "dpsda.private_histogram_rounds must equal "
            "num_samples_schedule_length - 1."
        )

    print(f"Repository root: {REPO_ROOT}")
    print(f"Dataset: {dataset['name']}")
    print(f"Columns: {len(columns)}")
    print(f"Synthetic rows: {synthetic_rows}")
    print(f"Epsilon: {epsilon}")
    print(f"Delta: {delta}")
    print(f"DPSDA private rounds: {private_rounds}")
    print("Step 4.1 configuration validation: OK")


def resolve_domain_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Load the prepared public domain spec once."""
    dataset = config["dataset"]
    with repo_path(dataset["domain_path"]).open(encoding="utf-8") as domain_file:
        return yaml.safe_load(domain_file)


def select_output_population(synthetic_data, requested_rows: int):
    """Select the final population that matches expected synthetic row count."""
    if len(synthetic_data.data_frame) == requested_rows:
        return synthetic_data

    if VARIATION_API_FOLD_ID_COLUMN_NAME not in synthetic_data.data_frame.columns:
        raise ValueError(
            "Fold marker column missing; cannot filter final-ranked population."
        )

    selected = synthetic_data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    selected_rows = len(selected.data_frame)
    if selected_rows != requested_rows:
        raise ValueError(
            "Unexpected filtered population size from DPSDA output. "
            f"Requested: {requested_rows}, raw: "
            f"{len(synthetic_data.data_frame)}, selected: {selected_rows}."
        )
    return selected


def extract_synthetic_dataframe(synthetic_data) -> pd.DataFrame:
    """Convert DPSDA synthetic data object into a flat dataframe."""
    feature_columns = list(synthetic_data.metadata.feature_columns)
    label_columns = list(synthetic_data.metadata.label_columns)
    feature_rows = pd.DataFrame(
        synthetic_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(),
        columns=feature_columns,
    )
    label_ids = synthetic_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
    for column in label_columns:
        feature_rows[column] = [
            synthetic_data.metadata.label_info[label_id].column_values[column]
            for label_id in label_ids
        ]
    return feature_rows


def postprocess_synthetic_dataframe(
    frame: pd.DataFrame,
    config: dict[str, Any],
    domain: dict[str, Any],
) -> pd.DataFrame:
    """Project to configured schema, cast numeric values, and clip domains."""
    dataset = config["dataset"]

    ordered_columns = list(dataset["columns"])
    expected = set(ordered_columns)
    actual = set(frame.columns)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Synthetic output columns do not match expected schema.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    canonical_numeric_type = dataset.get("canonical_numeric_type", "integer")
    if canonical_numeric_type != "integer":
        raise ValueError(
            "This benchmark adapter expects dataset.canonical_numeric_type = integer."
        )

    for numeric_column in dataset["numeric_columns"]:
        numeric_domain = domain[numeric_column]
        min_value = float(numeric_domain["min_value"])
        max_value = float(numeric_domain["max_value"])
        frame[numeric_column] = pd.to_numeric(
            frame[numeric_column], errors="coerce"
        ).round()
        frame[numeric_column] = frame[numeric_column].clip(
            lower=min_value, upper=max_value
        ).astype("int64")

    for categorical_column in dataset["categorical_columns"]:
        cat_domain = domain[categorical_column]
        possible_values = list(cat_domain.get("possible_values", []))
        if not possible_values:
            continue
        out_of_domain_index = int(cat_domain.get("out_of_domain_index", 0))
        fallback = possible_values[0]
        if 0 <= out_of_domain_index < len(possible_values):
            fallback = possible_values[out_of_domain_index]
        frame[categorical_column] = frame[categorical_column].where(
            frame[categorical_column].isin(possible_values), fallback
        )

    ordered = frame[ordered_columns].copy()
    if ordered.isna().any().any():
        null_columns = ordered.columns[ordered.isna().any()].tolist()
        raise ValueError(
            "Null values found in synthetic output columns: "
            f"{null_columns}"
        )
    return ordered


def default_output_path(config: dict[str, Any], seed: int, is_smoke: bool) -> Path:
    """Return default output path for a smoke or full run."""
    dataset_name = config["dataset"]["name"]
    run_mode = "smoke" if is_smoke else "full"
    return (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "raw"
        / "dpsda"
        / dataset_name
        / run_mode
        / f"seed-{seed}"
        / str(config["experiment"]["synthetic_rows"])
        / "synthetic.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DPSDA Tab-PE benchmark.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the benchmark YAML configuration.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs without running DPSDA.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 100-row, one-round smoke test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV path for synthetic samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to use for generation (defaults to the first config seed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")

    config = load_config(config_path)
    validate_config(config)
    seed = (
        args.seed
        if args.seed is not None
        else int(config["experiment"]["seeds"][0])
    )

    if args.validate_only:
        private_data, public_info, seed = prepare_dpsda_inputs(config, seed)
        runner = build_dpsda_runner(
            config=config,
            private_data=private_data,
            public_info=public_info,
        )

        categorical_count = sum(
            item["type"] == TabularColumnType.CATEGORICAL
            for item in public_info.values()
        )
        numerical_count = sum(
            item["type"] == TabularColumnType.FLOAT
            for item in public_info.values()
        )
        print(f"DPSDA seed: {seed}")
        print(f"DPSDA private rows: {len(private_data.data_frame)}")
        print(f"DPSDA public categorical features: {categorical_count}")
        print(f"DPSDA public numerical features: {numerical_count}")
        print("Step 4.2 DPSDA input preparation: OK")
        print(f"DPSDA runner class: {type(runner).__name__}")
        print("Step 4.3 DPSDA runner construction: OK")
        return
    if args.smoke:
        effective_config = make_smoke_config(config)
        run_mode = "smoke"
        expected_final_iteration = 1
    else:
        effective_config = config
        run_mode = "full"
        expected_final_iteration = None

    private_data, public_info, seed = prepare_dpsda_inputs(
        effective_config, seed
    )
    runner = build_dpsda_runner(
        config=effective_config,
        private_data=private_data,
        public_info=public_info,
    )

    num_rows = effective_config["experiment"]["synthetic_rows"]
    schedule_length = effective_config["dpsda"]["num_samples_schedule_length"]
    num_samples_schedule = [num_rows] * schedule_length
    label_fractions = effective_config["public_side_information"][
        "label_distribution"
    ]["fractions"]
    epsilon = effective_config["privacy"]["epsilon"]
    delta = effective_config["privacy"]["delta"]["value"]

    print(f"Starting Step 4.4 DPSDA {run_mode} run...")
    start_time = time.perf_counter()

    synthetic_data = runner.run(
        num_samples_schedule=num_samples_schedule,
        epsilon=epsilon,
        delta=delta,
        fraction_per_label_id=label_fractions,
        checkpoint_path=None,
        save_checkpoint=False,
    )

    elapsed_seconds = time.perf_counter() - start_time
    final_iteration = int(synthetic_data.metadata.iteration)
    if expected_final_iteration is None:
        expected_final_iteration = int(effective_config["dpsda"]["num_iterations"]) - 1

    if final_iteration != expected_final_iteration:
        raise ValueError(
            f"Expected final {run_mode} iteration {expected_final_iteration}, got "
            f"{final_iteration}."
        )

    selected = select_output_population(synthetic_data, num_rows)
    selected_rows = len(selected.data_frame)
    if selected_rows != num_rows:
        raise ValueError(
            f"{run_mode} requested {num_rows} rows but selected {selected_rows}."
        )

    raw_df = extract_synthetic_dataframe(selected)
    domain = resolve_domain_spec(effective_config)
    final_df = postprocess_synthetic_dataframe(raw_df, effective_config, domain)

    output_path = (
        repo_path(str(args.output))
        if args.output is not None
        else default_output_path(effective_config, seed, args.smoke)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"DPSDA {run_mode} seed: {seed}")
    print(f"DPSDA {run_mode} rows: {selected_rows}")
    print(f"DPSDA {run_mode} final iteration: {final_iteration}")
    print(f"DPSDA {run_mode} elapsed seconds: {elapsed_seconds:.2f}")
    print(f"DPSDA synthetic output saved: {output_path}")
    print(
        f"Step 4.4{'-smoke' if run_mode == 'smoke' else ' full'} DPSDA execution: OK"
    )
    print("Step 4.5 DPSDA synthetic output writing: OK")


if __name__ == "__main__":
    main()
