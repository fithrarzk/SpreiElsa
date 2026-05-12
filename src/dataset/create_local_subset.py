"""Create a small local subset of Intel Image Classification.

Run from the SpreiElsa project root:

    python src/dataset/create_local_subset.py --source ..\\dataset --output src\\dataset

The script keeps the full dataset outside the project and copies only a small,
deterministic sample per class for local CPU sanity checks.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")


@dataclass(frozen=True)
class SplitPlan:
    name: str
    source_split: str
    start_offset: int
    count_per_class: int


def _resolve_split_dir(source_root: Path, split_name: str) -> Path:
    """Handle Intel's common nested layout: seg_train/seg_train/<class>."""
    direct = source_root / split_name
    nested = direct / split_name

    if nested.is_dir():
        return nested
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find split directory for {split_name!r} under {source_root}")


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _copy_images(images: list[Path], destination: Path, overwrite: bool) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    for src in images:
        dst = destination / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied


def create_subset(
    source_root: Path,
    output_root: Path,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    classes: tuple[str, ...],
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    train_dir = _resolve_split_dir(source_root, "seg_train")
    test_dir = _resolve_split_dir(source_root, "seg_test")

    plans = (
        SplitPlan("train", "seg_train", 0, train_per_class),
        SplitPlan("val", "seg_train", train_per_class, val_per_class),
        SplitPlan("test", "seg_test", 0, test_per_class),
    )
    source_dirs = {"seg_train": train_dir, "seg_test": test_dir}
    summary: dict[str, dict[str, int]] = {}

    for plan in plans:
        summary[plan.name] = {}
        for class_name in classes:
            class_dir = source_dirs[plan.source_split] / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            available = _list_images(class_dir)
            selected = available[plan.start_offset : plan.start_offset + plan.count_per_class]
            copied = _copy_images(selected, output_root / plan.name / class_name, overwrite)
            summary[plan.name][class_name] = copied

    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../dataset"))
    parser.add_argument("--output", type=Path, default=Path("src/dataset"))
    parser.add_argument("--train-per-class", type=_positive_int, default=20)
    parser.add_argument("--val-per-class", type=_positive_int, default=5)
    parser.add_argument("--test-per-class", type=_positive_int, default=5)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--overwrite", action="store_true", help="Replace files that already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()

    summary = create_subset(
        source_root=source_root,
        output_root=output_root,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        classes=tuple(args.classes),
        overwrite=args.overwrite,
    )

    print(f"Created local subset in {output_root}")
    for split_name, split_counts in summary.items():
        total = sum(split_counts.values())
        class_counts = ", ".join(f"{name}: {count}" for name, count in split_counts.items())
        print(f"{split_name}: {total} images ({class_counts})")


if __name__ == "__main__":
    main()
