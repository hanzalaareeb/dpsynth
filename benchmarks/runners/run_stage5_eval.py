#!/usr/bin/env python3
"""Run Stage-5 evaluation for DPSynth and DPSDA outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run Stage-5 benchmark evaluation.")
  parser.add_argument(
      "--config",
      required=True,
      type=Path,
      help="Path to the benchmark YAML configuration.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      help="Seed override (defaults to the first seed in config).",
  )
  parser.add_argument(
      "--smoke",
      action="store_true",
      help="Run smoke-size outputs and metrics.",
  )
  parser.add_argument(
      "--skip-generation",
      action="store_true",
      help="Only run evaluation for existing synthetic files.",
  )
  parser.add_argument(
      "--dpsynth-python",
      type=str,
      default=sys.executable,
      help="Python interpreter used to run run_dpsynth.py.",
  )
  parser.add_argument(
      "--dpsda-python",
      type=str,
      default=sys.executable,
      help="Python interpreter used to run run_dpsda.py.",
  )
  parser.add_argument(
      "--eval-python",
      type=str,
      default=sys.executable,
      help="Python interpreter used to run run_tabular_eval.py.",
  )
  parser.add_argument(
      "--dpsynth-output",
      type=Path,
      help="Optional DPSynth synthetic CSV to evaluate.",
  )
  parser.add_argument(
      "--dpsda-output",
      type=Path,
      help="Optional DPSDA synthetic CSV to evaluate.",
  )
  parser.add_argument(
      "--report-dir",
      type=Path,
      default=Path("benchmarks/results/reports/stage5"),
      help="Directory for Stage-5 report outputs.",
  )
  return parser.parse_args()


def validate_config(config: dict[str, Any]) -> tuple[Path, Path]:
  """Validate benchmark files used by Stage-5."""
  dataset = config["dataset"]
  experiment = config["experiment"]
  train_path = repo_path(dataset["train_path"])
  test_path = repo_path(dataset["test_path"])
  if not train_path.exists():
    raise FileNotFoundError(f"Train path does not exist: {train_path}")
  if not test_path.exists():
    raise FileNotFoundError(f"Test path does not exist: {test_path}")
  if "seeds" not in experiment or not experiment["seeds"]:
    raise ValueError("experiment.seeds is missing or empty.")
  return train_path, test_path


def default_output_path(
    config: dict[str, Any],
    seed: int,
    method: str,
    run_mode: str,
) -> Path:
  """Return the expected raw output path for a method."""
  rows = 100 if run_mode == "smoke" else int(config["experiment"]["synthetic_rows"])
  return (
      REPO_ROOT
      / "benchmarks"
      / "results"
      / "raw"
      / method
      / config["dataset"]["name"]
      / run_mode
      / f"seed-{seed}"
      / str(rows)
      / "synthetic.csv"
  )


def run_generation_if_needed(
    method: str,
    config_path: Path,
    seed: int,
    synthetic_path: Path,
    smoke: bool,
    skip_generation: bool,
    runner_python: str,
) -> Path:
  """Generate synthetic output if it does not exist or generation is requested."""
  if not synthetic_path.exists():
    if skip_generation:
      raise FileNotFoundError(
          f"{method} synthetic output does not exist: {synthetic_path}"
      )
    cmd = [
        runner_python,
        str(REPO_ROOT / "benchmarks" / "runners" / f"run_{method}.py"),
        f"--config={config_path}",
        f"--seed={seed}",
        f"--output={synthetic_path}",
    ]
    if smoke:
      cmd.append("--smoke")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
  if not synthetic_path.exists():
    raise FileNotFoundError(
        f"{method} generation completed but output is missing: {synthetic_path}"
    )
  return synthetic_path


def run_eval(
    original_path: Path,
    synthetic_path: Path,
    report_path: Path,
    eval_python: str,
) -> None:
  """Run ``bin/run_tabular_eval.py`` for one synthetic candidate."""
  cmd = [
      eval_python,
      str(REPO_ROOT / "bin" / "run_tabular_eval.py"),
      f"--original_data_path={original_path}",
      f"--synthetic_data_path={synthetic_path}",
      f"--eval_report_path={report_path}",
      "--data_format=csv",
      "--use_beam=false",
  ]
  subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
  args = parse_args()
  config_path = args.config.resolve()
  if not config_path.is_file():
    raise FileNotFoundError(f"Configuration does not exist: {config_path}")
  config = load_config(config_path)
  _, test_path = validate_config(config)

  run_mode = "smoke" if args.smoke else "full"
  seed = args.seed if args.seed is not None else int(config["experiment"]["seeds"][0])

  methods = ["dpsynth", "dpsda"]
  report_root = repo_path(str(args.report_dir))
  rows = 100 if args.smoke else int(config["experiment"]["synthetic_rows"])
  run_timestamp = report_root / config["dataset"]["name"] / run_mode / f"seed-{seed}" / str(rows)
  run_timestamp.mkdir(parents=True, exist_ok=True)

  print(f"Stage-5 run mode: {run_mode}")
  print(f"Dataset: {config['dataset']['name']}")
  print(f"Seed: {seed}")
  print(f"Synthetic rows: {rows}")
  print(f"Original data for evaluation: {test_path}")

  method_outputs: dict[str, Path] = {}
  for method in methods:
    default_path = default_output_path(config, seed, method, run_mode)
    configured_output = (
        repo_path(str(args.dpsynth_output))
        if method == "dpsynth" and args.dpsynth_output is not None
        else repo_path(str(args.dpsda_output))
        if method == "dpsda" and args.dpsda_output is not None
        else default_path
    )
    generated_path = run_generation_if_needed(
        method=method,
        config_path=config_path,
        seed=seed,
        synthetic_path=configured_output,
        smoke=args.smoke,
        skip_generation=args.skip_generation,
        runner_python=args.dpsynth_python
        if method == "dpsynth"
        else args.dpsda_python,
    )
    method_outputs[method] = generated_path

  summaries = []
  for method, synthetic_path in method_outputs.items():
    report_path = run_timestamp / f"{method}_tabular_eval.pb"
    run_eval(
        original_path=test_path,
        synthetic_path=synthetic_path,
        report_path=report_path,
        eval_python=args.eval_python,
    )
    summaries.append(
        {
            "method": method,
            "synthetic_path": str(synthetic_path),
            "eval_report_path": str(report_path),
        }
    )
    print(f"Stage-5 {method} report: {report_path}")

  manifest_path = run_timestamp / "manifest.json"
  with manifest_path.open("w", encoding="utf-8") as manifest_file:
    json.dump(
        {
            "dataset": config["dataset"]["name"],
            "run_mode": run_mode,
            "seed": seed,
            "synthetic_rows": rows,
            "original_data_path": str(test_path),
            "results": summaries,
        },
        manifest_file,
        indent=2,
    )

  print("Step 5 metrics execution: OK")
  print(f"Stage-5 manifest written: {manifest_path}")


if __name__ == "__main__":
  main()
