"""Prepare Intel Image Classification layout for CNN training.

The Kaggle dataset is commonly laid out as:

    seg_train/seg_train/<class>/*.jpg
    seg_test/seg_test/<class>/*.jpg

The CNN trainer expects:

    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg

This script copies files into the expected layout without modifying the source
dataset. On Kaggle, write the output under ``/kaggle/working``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")


def has_prepared_layout(root: Path) -> bool:
    return all((root / split).is_dir() for split in ("train", "val", "test"))


def resolve_intel_split_dir(source_root: Path, split_name: str) -> Path:
    direct = source_root / split_name
    nested = direct / split_name
    if nested.is_dir():
        return nested
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"Cannot find {split_name!r} under {source_root}")


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_images(images: list[Path], destination: Path, overwrite: bool = False) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in images:
        dst = destination / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def prepare_intel_dataset(
    source_root: Path,
    output_root: Path,
    val_fraction: float = 0.15,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    overwrite: bool = False,
    max_train_per_class: int | None = None,
    max_val_per_class: int | None = None,
    max_test_per_class: int | None = None,
) -> dict[str, dict[str, int]]:
    """Copy Intel dataset into train/val/test class-folder layout."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    train_source = resolve_intel_split_dir(source_root, "seg_train")
    test_source = resolve_intel_split_dir(source_root, "seg_test")
    summary: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for class_name in classes:
        train_images = list_images(train_source / class_name)
        test_images = list_images(test_source / class_name)
        if not train_images:
            raise FileNotFoundError(f"No train images found for class {class_name}")
        if not test_images:
            raise FileNotFoundError(f"No test images found for class {class_name}")

        val_count = max(1, int(len(train_images) * val_fraction))
        val_images = train_images[-val_count:]
        train_images = train_images[:-val_count]

        if max_train_per_class is not None:
            train_images = train_images[:max_train_per_class]
        if max_val_per_class is not None:
            val_images = val_images[:max_val_per_class]
        if max_test_per_class is not None:
            test_images = test_images[:max_test_per_class]

        summary["train"][class_name] = copy_images(train_images, output_root / "train" / class_name, overwrite)
        summary["val"][class_name] = copy_images(val_images, output_root / "val" / class_name, overwrite)
        summary["test"][class_name] = copy_images(test_images, output_root / "test" / class_name, overwrite)

    return summary


def ensure_prepared_dataset(
    data_root: Path,
    prepared_root: Path,
    val_fraction: float = 0.15,
    overwrite: bool = False,
) -> Path:
    """Return a train/val/test dataset root, preparing Intel layout if needed."""
    if has_prepared_layout(data_root):
        return data_root
    prepare_intel_dataset(
        source_root=data_root,
        output_root=prepared_root,
        val_fraction=val_fraction,
        overwrite=overwrite,
    )
    return prepared_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-val-per-class", type=int)
    parser.add_argument("--max-test-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_intel_dataset(
        source_root=args.source.resolve(),
        output_root=args.output.resolve(),
        val_fraction=args.val_fraction,
        overwrite=args.overwrite,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        max_test_per_class=args.max_test_per_class,
    )
    print(f"Prepared dataset in {args.output.resolve()}")
    for split, counts in summary.items():
        total = sum(counts.values())
        detail = ", ".join(f"{class_name}: {count}" for class_name, count in counts.items())
        print(f"{split}: {total} images ({detail})")


if __name__ == "__main__":
    main()

