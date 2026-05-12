"""Train CNN experiments with Keras.

Local usage should run one tiny experiment only. Full ``--run-all`` training is
intended for Kaggle GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.cnn.experiments import CLASS_NAMES, all_shared_experiments, get_experiment
from src.cnn.keras_models import build_non_shared_cnn, build_shared_cnn


def _dataset(split_dir: Path, image_size: int, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=shuffle,
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def _evaluate_macro_f1(model: tf.keras.Model, dataset: tf.data.Dataset) -> float:
    y_true: list[int] = []
    y_pred: list[int] = []
    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        y_true.extend(np.asarray(y_batch).astype(int).tolist())
        y_pred.extend(np.argmax(probs, axis=1).astype(int).tolist())
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def train_one(config, data_root: Path, output_dir: Path, epochs: int, batch_size: int, non_shared: bool = False) -> Path:
    experiment_name = f"{'non_shared' if non_shared else 'shared'}_{config.experiment_id}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _dataset(data_root / "train", config.image_size, batch_size, shuffle=True)
    val_ds = _dataset(data_root / "val", config.image_size, batch_size, shuffle=False)
    test_ds = _dataset(data_root / "test", config.image_size, batch_size, shuffle=False)

    model = build_non_shared_cnn(config) if non_shared else build_shared_cnn(config)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    macro_f1 = _evaluate_macro_f1(model, test_ds)

    model.save(experiment_dir / "model.keras")
    model.save_weights(experiment_dir / "weights.weights.h5")
    _write_json(experiment_dir / "history.json", {key: [float(v) for v in values] for key, values in history.history.items()})
    _write_json(experiment_dir / "config.json", config.to_dict() | {"non_shared": non_shared})
    _write_json(
        experiment_dir / "metrics.json",
        {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "macro_f1": macro_f1,
            "param_count": int(model.count_params()),
            "class_names": CLASS_NAMES,
        },
    )
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("src/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--experiment-id", default="d1_f16_k3_max")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--non-shared", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = all_shared_experiments(args.image_size) if args.run_all else [get_experiment(args.experiment_id, args.image_size)]
    for config in configs:
        train_one(config, args.data_root, args.output_dir, args.epochs, args.batch_size, non_shared=args.non_shared)


if __name__ == "__main__":
    main()

