from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history_path: Path, output_path: Path) -> None:
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history.get("loss", []), label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_experiments(output_root: Path, summary_path: Path) -> list[dict]:
    rows: list[dict] = []
    for experiment_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        config_path = experiment_dir / "config.json"
        metrics_path = experiment_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "experiment": experiment_dir.name,
                "conv_depth": config.get("conv_depth"),
                "filters_base": config.get("filters_base"),
                "kernel_size": config.get("kernel_size"),
                "pooling": config.get("pooling"),
                "non_shared": config.get("non_shared", False),
                "macro_f1": metrics.get("macro_f1"),
                "test_accuracy": metrics.get("test_accuracy"),
                "param_count": metrics.get("param_count"),
            }
        )
        history_path = experiment_dir / "history.json"
        if history_path.exists():
            plot_history(history_path, experiment_dir / "loss.png")

    rows.sort(key=lambda row: row.get("macro_f1") or -1, reverse=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["experiment"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--summary-path", type=Path, default=Path("outputs/cnn/summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_experiments(args.output_root, args.summary_path)
    print(f"Wrote {len(rows)} rows to {args.summary_path}")


if __name__ == "__main__":
    main()

