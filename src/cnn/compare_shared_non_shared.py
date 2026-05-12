"""Build a compact shared vs non-shared comparison from experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare(shared_dir: Path, non_shared_dir: Path, output_path: Path) -> dict:
    shared_metrics = _read_json(shared_dir / "metrics.json")
    non_shared_metrics = _read_json(non_shared_dir / "metrics.json")
    rows = [
        {
            "model_type": "shared",
            "experiment": shared_dir.name,
            "macro_f1": shared_metrics.get("macro_f1"),
            "test_accuracy": shared_metrics.get("test_accuracy"),
            "param_count": shared_metrics.get("param_count"),
        },
        {
            "model_type": "non_shared",
            "experiment": non_shared_dir.name,
            "macro_f1": non_shared_metrics.get("macro_f1"),
            "test_accuracy": non_shared_metrics.get("test_accuracy"),
            "param_count": non_shared_metrics.get("param_count"),
        },
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump({"rows": rows}, handle, indent=2)
    else:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {"rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--non-shared-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(args.shared_dir, args.non_shared_dir, args.output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

