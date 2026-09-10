#!/usr/bin/env python3
"""Run DPSynth baseline benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import dpsynth
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MECHANISM_MAP = {
    "mst": dpsynth.discrete_mechanisms.MSTConfig,
    "aim": dpsynth.discrete_mechanisms.AIMConfig,
    "independent": dpsynth.discrete_mechanisms.IndependentConfig,
    "aim_gdp": dpsynth.discrete_mechanisms.AIMGDPConfig,
}


def repo_path(value: str) -> Path:
  """Resolve a benchmark-relative path to an absolute path."""
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


def normalize_synthetic_rows(
    frame: pd.DataFrame,
    target_rows: int,
    seed: int,
) -> pd.DataFrame:
  """Resize synthetic rows deterministically to the requested count."""
  current_rows = len(frame)
  if current_rows == target_rows:
    return frame

  rng = np.random.default_rng(seed)
  if current_rows > target_rows:
    selected_indices = rng.choice(current_rows, size=target_rows, replace=False)
    return frame.iloc[selected_indices].reset_index(drop=True)

  needed_rows = target_rows - current_rows
  sampled_indices = rng.choice(current_rows, size=needed_rows, replace=True)
  sampled = frame.iloc[sampled_indices]
  return pd.concat([frame, sampled], ignore_index=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run DPSynth baseline benchmark.")
  parser.add_argument(
      "--config",
      required=True,
      type=Path,
      help="Path to the benchmark YAML configuration.",
  )
  parser.add_argument(
      "--validate-only",
      action="store_true",
      help="Validate inputs without running DPSynth.",
  )
  parser.add_argument(
      "--smoke",
      action="store_true",
      help="Run a reduced-size smoke run for quick verification.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      help="Seed to use for sampling and normalization.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      help="Optional output CSV path for synthetic samples.",
  )
  return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
  """Validate benchmark files and experiment parameters."""
  dataset = config["dataset"]
  privacy = config["privacy"]
  experiment = config["experiment"]
  dpsynth_config = config.get("dpsynth", {})

  required_paths = {
      "train": repo_path(dataset["train_path"]),
      "test": repo_path(dataset["test_path"]),
      "domain": repo_path(dataset["domain_path"]),
      "manifest": repo_path(dataset["manifest"]),
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
        f"Actual: {test_header}"
    )

  epsilon = privacy["epsilon"]
  delta = privacy["delta"]["value"]
  if epsilon <= 0 or delta <= 0:
    raise ValueError("privacy.epsilon and privacy.delta.value must be positive.")

  synthetic_rows = int(experiment["synthetic_rows"])
  if synthetic_rows <= 0:
    raise ValueError("experiment.synthetic_rows must be a positive integer.")

  mechanism = str(dpsynth_config.get("mechanism", "aim"))
  if mechanism not in {"mst", "aim", "independent", "aim_gdp"}:
    raise ValueError(
        "Unsupported dpsynth mechanism. Use one of: mst, aim, independent, aim_gdp."
    )

  seeds = experiment["seeds"]
  if not seeds or not isinstance(seeds, list):
    raise ValueError("experiment.seeds must be a non-empty list.")

  print(f"Repository root: {REPO_ROOT}")
  print(f"Dataset: {dataset['name']}")
  print(f"Columns: {len(columns)}")
  print(f"Synthetic rows: {synthetic_rows}")
  print(f"Epsilon: {epsilon}")
  print(f"Delta: {delta}")
  print(f"DPSynth mechanism: {mechanism}")
  print("Step 5.1 configuration validation: OK")


def default_output_path(
    config: dict[str, Any],
    seed: int,
    is_smoke: bool,
) -> Path:
  """Return the default output path for a smoke or full run."""
  dataset_name = config["dataset"]["name"]
  run_mode = "smoke" if is_smoke else "full"
  return (
      REPO_ROOT
      / "benchmarks"
      / "results"
      / "raw"
      / "dpsynth"
      / dataset_name
      / run_mode
      / f"seed-{seed}"
      / str(config["experiment"]["synthetic_rows"])
      / "synthetic.csv"
  )


def run_dpsynth_in_process(
    dataset_path: Path,
    domain_path: Path,
    epsilon: float,
    delta: float,
    mechanism: str,
    seed: int,
) -> pd.DataFrame:
  """Generate synthetic data directly via the DPSynth Python API."""
  mechanism_name = mechanism.lower()
  mechanism_factory = MECHANISM_MAP.get(mechanism_name)
  if mechanism_factory is None:
    raise ValueError(
        f"Unsupported dpsynth mechanism: {mechanism}."
    )

  domains = dpsynth.domain.from_yaml_file(str(domain_path))
  mechanism_config = mechanism_factory()
  synthesizer = dpsynth.TabularConfig(
      domains=domains,
      discrete_mechanism=mechanism_config,
  )
  calibrated = synthesizer.calibrate(epsilon=epsilon, delta=delta)
  data = pd.read_csv(dataset_path)
  result = calibrated(np.random.default_rng(seed), data)
  return result.synthetic_data


def main() -> None:
  args = parse_args()
  config_path = args.config.resolve()
  if not config_path.is_file():
    raise FileNotFoundError(f"Configuration does not exist: {config_path}")

  config = load_config(config_path)
  validate_config(config)

  dataset = config["dataset"]
  privacy = config["privacy"]
  experiment = config["experiment"]
  dpsynth_config = config.get("dpsynth", {})
  mechanism = str(dpsynth_config.get("mechanism", "aim"))

  run_mode = "smoke" if args.smoke else "full"
  seed = args.seed if args.seed is not None else int(experiment["seeds"][0])
  synthetic_rows = 100 if args.smoke else int(experiment["synthetic_rows"])
  effective = dict(config)
  effective["experiment"] = dict(experiment)
  effective["experiment"]["synthetic_rows"] = synthetic_rows

  print(
      f"DPSynth seed: {seed}\n"
      f"DPSynth rows before normalization target: {synthetic_rows}"
  )

  if args.validate_only:
    print("Step 5.2 DPSynth input preparation: OK")
    return

  dataset_path = repo_path(dataset["train_path"])
  domain_path = repo_path(dataset["domain_path"])
  output_path = (
      repo_path(str(args.output))
      if args.output is not None
      else default_output_path(effective, seed, args.smoke)
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)

  synthetic_df = run_dpsynth_in_process(
      dataset_path=dataset_path,
      domain_path=domain_path,
      epsilon=float(privacy["epsilon"]),
      delta=float(privacy["delta"]["value"]),
      mechanism=mechanism,
      seed=seed,
  )
  final_df = normalize_synthetic_rows(synthetic_df, synthetic_rows, seed=seed)
  final_df.to_csv(output_path, index=False)

  print(f"DPSynth {run_mode} synthetic rows (normalized): {len(final_df)}")
  print(f"DPSynth synthetic output saved: {output_path}")
  print(f"Step 5.3 DPSynth {run_mode} execution: OK")


if __name__ == "__main__":
  main()
