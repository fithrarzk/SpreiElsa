from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cnn.experiments import all_shared_experiments
from src.cnn.plotting import summarize_experiments
from src.cnn.train_cnn import train_one
from src.dataset.prepare_intel_dataset import ensure_prepared_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path containing train/val/test or raw Intel seg_train/seg_test layout.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs/cnn"))
    parser.add_argument("--prepared-data-root", type=Path, default=Path("/kaggle/working/intel_prepared"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = ensure_prepared_dataset(
        data_root=args.data_root,
        prepared_root=args.prepared_data_root,
        val_fraction=args.val_fraction,
    )
    for config in all_shared_experiments(args.image_size):
        train_one(config, data_root, args.output_dir, args.epochs, args.batch_size, non_shared=False)

    rows = summarize_experiments(args.output_dir, args.output_dir / "summary.csv")
    if not rows:
        raise RuntimeError("No shared experiment metrics found")

    best_id = rows[0]["experiment"].replace("shared_", "")
    best_config = next(config for config in all_shared_experiments(args.image_size) if config.experiment_id == best_id)
    train_one(best_config, data_root, args.output_dir, args.epochs, args.batch_size, non_shared=True)
    summarize_experiments(args.output_dir, args.output_dir / "summary.csv")

    with (args.output_dir / "best_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_shared_experiment": best_id, "macro_f1": rows[0]["macro_f1"]}, handle, indent=2)


if __name__ == "__main__":
    main()
