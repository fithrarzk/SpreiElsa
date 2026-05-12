"""Evaluate saved Keras CNN artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score

from src.cnn.experiments import CLASS_NAMES
from src.cnn.keras_models import SimpleLocallyConnected2D


def _dataset(split_dir: Path, image_size: int, batch_size: int) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=False,
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)


def evaluate_model(model_path: Path, data_root: Path, output_dir: Path, image_size: int = 64, batch_size: int = 32) -> dict:
    model = tf.keras.models.load_model(model_path, custom_objects={"SimpleLocallyConnected2D": SimpleLocallyConnected2D})
    test_ds = _dataset(data_root / "test", image_size, batch_size)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true.extend(np.asarray(y_batch).astype(int).tolist())
        y_pred.extend(np.argmax(probs, axis=1).astype(int).tolist())
        y_prob.extend(probs.tolist())

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "param_count": int(model.count_params()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "class_names": CLASS_NAMES,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "y_true", "y_pred", "probabilities"])
        for index, (true, pred, probs) in enumerate(zip(y_true, y_pred, y_prob)):
            writer.writerow([index, true, pred, json.dumps(probs)])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("src/dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_model(args.model_path, args.data_root, args.output_dir, args.image_size, args.batch_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

