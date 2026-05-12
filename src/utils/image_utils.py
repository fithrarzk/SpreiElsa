"""Image utilities for the CNN and image-captioning pipelines.

All preprocessing here uses Pillow and NumPy, not Keras preprocessing layers.
Images are returned in channel-last RGB format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(
    path: str | Path,
    image_size: tuple[int, int] = (64, 64),
    normalize: bool = True,
) -> np.ndarray:
    """Load one RGB image.

    Args:
        path: Image file path.
        image_size: Target size as ``(height, width)``.
        normalize: If true, return float32 values in ``[0, 1]``. Otherwise
            return uint8-like float32 values in ``[0, 255]``.

    Returns:
        Array with shape ``(height, width, 3)``.
    """
    height, width = image_size
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height))
        array = np.asarray(image, dtype=np.float32)
    if normalize:
        array = array / 255.0
    return array


def load_images(
    paths: Sequence[str | Path],
    image_size: tuple[int, int] = (64, 64),
    normalize: bool = True,
) -> np.ndarray:
    """Load a batch of images into ``(N, H, W, C)`` format."""
    images = [load_image(path, image_size=image_size, normalize=normalize) for path in paths]
    if not images:
        height, width = image_size
        return np.empty((0, height, width, 3), dtype=np.float32)
    return np.stack(images).astype(np.float32, copy=False)


def list_image_paths(
    root: str | Path,
    class_names: Sequence[str] | None = None,
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Scan a class-folder split.

    Args:
        root: Split root with children like ``buildings/*.jpg``.
        class_names: Optional fixed class order. If omitted, class folders are
            sorted alphabetically.

    Returns:
        ``(paths, labels, class_names)`` where labels are integer class ids.
    """
    split_root = Path(root)
    if not split_root.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_root}")

    classes = list(class_names) if class_names is not None else sorted(
        path.name for path in split_root.iterdir() if path.is_dir()
    )
    class_to_index = {name: index for index, name in enumerate(classes)}
    paths: list[Path] = []
    labels: list[int] = []

    for class_name in classes:
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(image_path)
                labels.append(class_to_index[class_name])

    return paths, np.asarray(labels, dtype=np.int64), classes


def iter_image_paths(root: str | Path) -> Iterable[tuple[Path, str]]:
    """Yield ``(image_path, class_name)`` from a class-folder split."""
    paths, labels, classes = list_image_paths(root)
    for path, label in zip(paths, labels):
        yield path, classes[int(label)]


def load_classification_split(
    root: str | Path,
    image_size: tuple[int, int] = (64, 64),
    class_names: Sequence[str] | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a small class-folder split fully into memory.

    Intended for local subset sanity checks. Full training should use Keras
    streaming datasets.
    """
    paths, labels, classes = list_image_paths(root, class_names=class_names)
    images = load_images(paths, image_size=image_size, normalize=normalize)
    return images, labels, classes


def save_features(features: np.ndarray, path: str | Path) -> None:
    """Save feature vectors to ``.npy``."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)


def load_features(path: str | Path) -> np.ndarray:
    """Load feature vectors saved by :func:`save_features`."""
    return np.load(Path(path), allow_pickle=False)


def extract_features_with_keras_encoder(
    image_paths: Sequence[str | Path],
    encoder,
    output_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> np.ndarray:
    """Extract image features with a frozen Keras encoder and save ``.npy``.

    The encoder is expected to accept normalized RGB arrays with shape
    ``(N, H, W, 3)`` and return feature vectors or feature maps.
    """
    batches: list[np.ndarray] = []
    paths = list(image_paths)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_images(batch_paths, image_size=image_size, normalize=True)
        features = encoder.predict(batch, verbose=0)
        batches.append(np.asarray(features))

    if batches:
        all_features = np.concatenate(batches, axis=0)
    else:
        all_features = np.empty((0,), dtype=np.float32)

    save_features(all_features, output_path)
    return all_features

