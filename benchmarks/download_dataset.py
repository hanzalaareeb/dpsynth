#!/usr/bin/env python3
"""Download and verify a benchmark dataset from its checked-in manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Download files declared by a benchmark dataset manifest."
  )
  parser.add_argument("manifest", type=Path)
  return parser.parse_args()


def download_file(url: str, output_path: Path) -> bytes:
  """Download a URL and atomically install it at output_path."""
  with urllib.request.urlopen(url) as response:
    content = response.read()
  temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
  temporary_path.write_bytes(content)
  temporary_path.replace(output_path)
  return content


def csv_data_rows(content: bytes) -> int:
  """Return the number of data rows in a UTF-8 CSV byte string."""
  lines = content.decode("utf-8").splitlines()
  return max(0, sum(1 for _ in csv.reader(lines)) - 1)


def main() -> None:
  args = parse_args()
  manifest_path = args.manifest.resolve()
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  output_directory = REPO_ROOT / "benchmarks" / "data" / manifest["name"]
  output_directory.mkdir(parents=True, exist_ok=True)

  for filename, specification in manifest["files"].items():
    if Path(filename).name != filename:
      raise ValueError(f"Manifest filename must not contain a path: {filename}")
    output_path = output_directory / filename
    content = download_file(specification["url"], output_path)
    digest = hashlib.sha256(content).hexdigest()
    if digest != specification["sha256"]:
      output_path.unlink(missing_ok=True)
      raise ValueError(
          f"SHA-256 mismatch for {filename}: expected "
          f"{specification['sha256']}, received {digest}"
      )
    if len(content) != specification["bytes"]:
      output_path.unlink(missing_ok=True)
      raise ValueError(
          f"Byte count mismatch for {filename}: expected "
          f"{specification['bytes']}, received {len(content)}"
      )
    if filename.endswith(".csv"):
      rows = csv_data_rows(content)
      if rows != specification["rows"]:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"Row count mismatch for {filename}: expected "
            f"{specification['rows']}, received {rows}"
        )
    print(f"Verified {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
  main()
