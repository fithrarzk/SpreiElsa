"""Train RNN/LSTM decoder models for Flickr8k."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from text_utils import load_vocab

sys.path.insert(0, os.path.dirname(__file__))
from decoder_model import build_dataset, build_decoder_model, masked_accuracy, masked_loss
from feature_extraction import load_features


def _load_splits(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    return {
        "train": splits.get("train", []),
        "val": splits.get("val", []),
        "test": splits.get("test", []),
    }


@dataclass(frozen=True)
class DecoderExperimentConfig:
    decoder_type: str
    n_layers: int
    hidden_size: int
    embed_dim: int
    max_seq_len: int
    feature_dim: int
    learning_rate: float

    @property
    def experiment_id(self) -> str:
        return f"{self.decoder_type}_l{self.n_layers}_h{self.hidden_size}_e{self.embed_dim}_m{self.max_seq_len}"

    def to_dict(self) -> dict:
        return asdict(self) | {"experiment_id": self.experiment_id}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _build_model(config: DecoderExperimentConfig, vocab_size: int) -> tf.keras.Model:
    model = build_decoder_model(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        decoder_type=config.decoder_type,
        feature_dim=config.feature_dim,
        max_seq_len=config.max_seq_len,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=masked_loss,
        metrics=[masked_accuracy],
    )
    return model


def train_one(
    config: DecoderExperimentConfig,
    captions: dict,
    splits: dict[str, list[str]],
    features: dict,
    word2idx: dict,
    output_root: Path,
    batch_size: int,
    epochs: int,
    seed: int,
) -> Path:
    experiment_dir = output_root / config.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(
        image_names=splits["train"],
        all_captions=captions,
        features=features,
        word2idx=word2idx,
        max_seq_len=config.max_seq_len,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = build_dataset(
        image_names=splits["val"],
        all_captions=captions,
        features=features,
        word2idx=word2idx,
        max_seq_len=config.max_seq_len,
        batch_size=batch_size,
        shuffle=False,
    )

    model = _build_model(config, vocab_size=len(word2idx))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save(experiment_dir / "model.keras")
    model.save_weights(experiment_dir / "weights.weights.h5")
    _write_json(experiment_dir / "history.json", {key: [float(v) for v in values] for key, values in history.history.items()})
    _write_json(experiment_dir / "config.json", config.to_dict())
    _write_json(
        experiment_dir / "metrics.json",
        {
            "train_loss": float(history.history.get("loss", [0.0])[-1]),
            "val_loss": float(history.history.get("val_loss", [0.0])[-1]),
            "param_count": int(model.count_params()),
        },
    )
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-file", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rnn_lstm"))
    parser.add_argument("--decoder-type", choices=["rnn", "lstm"], default="lstm")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=35)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-grid", action="store_true")
    parser.add_argument("--layers-grid", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--hidden-grid", nargs="+", type=int, default=[128, 512])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from text_utils import load_flickr8k_captions

    captions = load_flickr8k_captions(str(args.captions_file))
    splits = _load_splits(args.splits_json)
    word2idx, _ = load_vocab(args.vocab_path)
    features = load_features(str(args.features_path))

    output_root = args.output_dir / args.decoder_type
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_grid:
        configs = [
            DecoderExperimentConfig(
                decoder_type=args.decoder_type,
                n_layers=n_layers,
                hidden_size=hidden_size,
                embed_dim=args.embed_dim,
                max_seq_len=args.max_seq_len,
                feature_dim=args.feature_dim,
                learning_rate=args.learning_rate,
            )
            for n_layers, hidden_size in product(args.layers_grid, args.hidden_grid)
        ]
    else:
        configs = [
            DecoderExperimentConfig(
                decoder_type=args.decoder_type,
                n_layers=args.n_layers,
                hidden_size=args.hidden_size,
                embed_dim=args.embed_dim,
                max_seq_len=args.max_seq_len,
                feature_dim=args.feature_dim,
                learning_rate=args.learning_rate,
            )
        ]

    for config in configs:
        train_one(
            config=config,
            captions=captions,
            splits=splits,
            features=features,
            word2idx=word2idx,
            output_root=output_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
