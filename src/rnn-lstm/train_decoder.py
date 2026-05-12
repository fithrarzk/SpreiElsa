"""Train RNN/LSTM decoder models for Flickr8k."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
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
    injection_method: str

    @property
    def experiment_id(self) -> str:
        return (
            f"{self.decoder_type}_{self.injection_method}"
            f"_l{self.n_layers}_h{self.hidden_size}_e{self.embed_dim}_m{self.max_seq_len}"
        )

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
        injection_method=config.injection_method,
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
        injection_method=config.injection_method,
    )
    val_ds = build_dataset(
        image_names=splits["val"],
        all_captions=captions,
        features=features,
        word2idx=word2idx,
        max_seq_len=config.max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        injection_method=config.injection_method,
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


