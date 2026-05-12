"""Preprocess Flickr8k captions and splits for training."""

from __future__ import annotations

import json
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from text_utils import (
    build_vocabulary,
    caption_length_stats,
    load_flickr8k_captions,
    load_split_names,
    save_vocab,
)


def _read_split(path: Path) -> list[str]:
    names = load_split_names(path)
    return [name.strip() for name in names if name.strip()]


def _filter_captions(captions: dict, image_names: list[str]) -> dict:
    return {name: captions[name] for name in image_names if name in captions}


def _flatten_captions(captions_dict: dict) -> list[str]:
    captions = []
    for items in captions_dict.values():
        captions.extend(items)
    return captions


def preprocess(
    captions_file: Path,
    train_split: Path,
    val_split: Path,
    test_split: Path,
    output_dir: Path,
    min_freq: int,
) -> dict:
    captions = load_flickr8k_captions(str(captions_file))
    train_names = _read_split(train_split)
    val_names = _read_split(val_split)
    test_names = _read_split(test_split)

    train_caps = _filter_captions(captions, train_names)
    val_caps = _filter_captions(captions, val_names)
    test_caps = _filter_captions(captions, test_names)

    word2idx, idx2word = build_vocabulary(train_caps, min_freq=min_freq)
    stats = caption_length_stats(_flatten_captions(train_caps), word2idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(word2idx, idx2word, output_dir / "vocab.json")

    splits = {
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    summary = {
        "total_images": len(captions),
        "train_images": len(train_names),
        "val_images": len(val_names),
        "test_images": len(test_names),
        "train_images_with_captions": len(train_caps),
        "val_images_with_captions": len(val_caps),
        "test_images_with_captions": len(test_caps),
        "vocab_size": len(word2idx),
        "caption_length_stats": stats,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


