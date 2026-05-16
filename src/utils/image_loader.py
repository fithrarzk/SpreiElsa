from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .image_utils import (
    IMAGE_EXTENSIONS,
    iter_image_paths,
    load_classification_split,
    load_image,
    load_images,
)


def load_split(
    root: str | Path,
    image_size: tuple[int, int] = (64, 64),
    class_names: Sequence[str] | None = None,
):
    """Compatibility wrapper for older code."""
    return load_classification_split(root, image_size=image_size, class_names=class_names)
