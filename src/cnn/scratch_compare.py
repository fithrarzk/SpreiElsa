from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.cnn.cnn_scratch.weight_io import keras_to_scratch_model
from src.cnn.experiments import CLASS_NAMES
from src.cnn.keras_models import SimpleLocallyConnected2D
from src.utils.image_utils import load_classification_split


def compare_keras_scratch(
    model_path: Path,
    split_root: Path,
    output_path: Path,
    image_size: int = 64,
    max_samples: int = 30,
) -> dict:
    model = tf.keras.models.load_model(model_path, custom_objects={"SimpleLocallyConnected2D": SimpleLocallyConnected2D})
    scratch = keras_to_scratch_model(model)
    x, y, _ = load_classification_split(split_root, image_size=(image_size, image_size), class_names=CLASS_NAMES)
    x = x[:max_samples]
    y = y[:max_samples]

    keras_probs = model.predict(x, verbose=0)
    scratch_probs = scratch.predict(x)
    keras_pred = np.argmax(keras_probs, axis=1)
    scratch_pred = np.argmax(scratch_probs, axis=1)
    metrics = {
        "samples": int(len(y)),
        "keras_macro_f1": float(f1_score(y, keras_pred, average="macro", zero_division=0)),
        "scratch_macro_f1": float(f1_score(y, scratch_pred, average="macro", zero_division=0)),
        "max_abs_probability_diff": float(np.max(np.abs(keras_probs - scratch_probs))) if len(y) else 0.0,
        "prediction_match_rate": float(np.mean(keras_pred == scratch_pred)) if len(y) else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=Path("src/dataset/test"))
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compare_keras_scratch(args.model_path, args.split_root, args.output_path, args.image_size, args.max_samples)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

