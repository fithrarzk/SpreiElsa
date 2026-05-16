"""Evaluate captioning models with BLEU-4 and METEOR."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from text_utils import clean_caption, decode_caption, load_flickr8k_captions, load_vocab

sys.path.insert(0, os.path.dirname(__file__))
from caption_model import CaptionModel
from feature_extraction import load_features
from metrics import bleu_score, meteor_score, tokenize


def _load_splits(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    return {
        "train": splits.get("train", []),
        "val": splits.get("val", []),
        "test": splits.get("test", []),
    }


def _filter_image_names(names: list[str], captions: dict, features: dict) -> list[str]:
    return [name for name in names if name in captions and name in features]


def _greedy_decode_keras(model, image_feature: np.ndarray, word2idx: dict, idx2word: dict,
                         max_len: int) -> str:
    pad_idx   = word2idx.get("<pad>", 0)
    start_idx = word2idx.get("<start>", 1)
    end_idx   = word2idx.get("<end>", 2)

    caption_len = int(model.inputs[1].shape[1])
    output_len  = int(model.outputs[0].shape[1])
    steps = min(max_len, output_len)

    tokens = [start_idx]
    for step in range(steps):
        caption_input = np.full((1, caption_len), pad_idx, dtype=np.int32)
        cap_len = min(len(tokens), caption_len)
        caption_input[0, :cap_len] = tokens[:cap_len]
        preds = model.predict([image_feature[np.newaxis], caption_input], verbose=0)
        next_token = int(np.argmax(preds[0, step]))
        if next_token in (end_idx, pad_idx):
            break
        tokens.append(next_token)

    return decode_caption(tokens, idx2word)


def _beam_search_keras(
    model,
    image_feature: np.ndarray,
    word2idx: dict,
    idx2word: dict,
    max_len: int,
    beam_size: int,
    length_penalty: float,
) -> str:
    pad_idx = word2idx.get("<pad>", 0)
    start_idx = word2idx.get("<start>", 1)
    end_idx = word2idx.get("<end>", 2)

    caption_len = int(model.inputs[1].shape[1])
    output_len = int(model.outputs[0].shape[1])
    steps = min(max_len, output_len)

    def score_norm(score, length):
        if length_penalty <= 0:
            return score
        return score / (length ** length_penalty)

    beams = [([start_idx], 0.0)]
    for step in range(steps):
        candidates = []
        for tokens, score in beams:
            last = tokens[-1]
            if last in (end_idx, pad_idx):
                candidates.append((tokens, score))
                continue
            caption_input = np.full((1, caption_len), pad_idx, dtype=np.int32)
            cap_len = min(len(tokens), caption_len)
            caption_input[0, :cap_len] = tokens[:cap_len]
            preds = model.predict([image_feature[np.newaxis], caption_input], verbose=0)
            probs = preds[0, step]
            top_idx = np.argsort(probs)[-beam_size:][::-1]
            for idx in top_idx:
                prob = float(probs[int(idx)])
                new_score = score + float(np.log(max(prob, 1e-9)))
                candidates.append((tokens + [int(idx)], new_score))

        candidates.sort(key=lambda item: score_norm(item[1], len(item[0])), reverse=True)
        beams = candidates[:beam_size]
        if all(tokens[-1] in (end_idx, pad_idx) for tokens, _ in beams):
            break

    best_tokens = max(beams, key=lambda item: score_norm(item[1], len(item[0])))[0]
    return decode_caption(best_tokens, idx2word)


def _evaluate_single(
    image_name: str,
    candidate: str,
    references: list[str],
) -> dict:
    candidate_tokens = tokenize(candidate)
    reference_tokens = [tokenize(ref) for ref in references]
    bleu4 = bleu_score(candidate_tokens, reference_tokens, max_n=4)
    meteor = meteor_score(candidate_tokens, reference_tokens)
    return {
        "image": image_name,
        "candidate": candidate,
        "references": references,
        "bleu4": bleu4,
        "meteor": meteor,
    }


def evaluate(
    model,
    captions: dict,
    image_names: list[str],
    features: dict,
    word2idx: dict,
    idx2word: dict,
    decoder_type: str,
    max_len: int,
    mode: str,
    seed: int,
    decoding: str,
    beam_size: int,
    length_penalty: float,
    injection_method: str = "pre",
) -> dict:
    rng = random.Random(seed)
    results = []
    for image_name in image_names:
        references = captions.get(image_name, [])
        if not references:
            continue
        feature = np.asarray(features[image_name], dtype=np.float32)
        if mode == "keras":
            if decoding == "beam":
                candidate = _beam_search_keras(
                    model, feature, word2idx, idx2word,
                    max_len=max_len, beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                candidate = _greedy_decode_keras(
                    model, feature, word2idx, idx2word,
                    max_len=max_len,
                )
        else:
            if decoding == "beam":
                candidate = model.generate_caption_beam(
                    feature, max_len=max_len, beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                candidate = model.generate_caption(feature, max_len=max_len)
        results.append(_evaluate_single(image_name, candidate, references))

    rng.shuffle(results)
    bleu_avg = float(np.mean([row["bleu4"] for row in results])) if results else 0.0
    meteor_avg = float(np.mean([row["meteor"] for row in results])) if results else 0.0

    return {
        "metrics": {
            "bleu4": bleu_avg,
            "meteor": meteor_avg,
            "samples": len(results),
            "mode": mode,
            "decoder_type": decoder_type,
        },
        "details": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-file", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rnn_lstm/eval"))
    parser.add_argument("--decoder-type", choices=["rnn", "lstm"], default="lstm")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["keras", "scratch", "both"], default="keras")
    parser.add_argument("--injection", choices=["pre", "init"], default="pre")
    parser.add_argument("--decoding", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--length-penalty", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    captions = load_flickr8k_captions(str(args.captions_file))
    splits = _load_splits(args.splits_json)
    word2idx, idx2word = load_vocab(args.vocab_path)
    features = load_features(str(args.features_path))

    image_names = _filter_image_names(splits[args.split], captions, features)
    if args.max_samples:
        image_names = image_names[: args.max_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("keras", "both"):
        keras_model = tf.keras.models.load_model(args.model_path, compile=False)
        result = evaluate(
            model=keras_model,
            captions=captions,
            image_names=image_names,
            features=features,
            word2idx=word2idx,
            idx2word=idx2word,
            decoder_type=args.decoder_type,
            max_len=args.max_len,
            mode="keras",
            seed=args.seed,
            decoding=args.decoding,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            injection_method=args.injection,
        )
        with (args.output_dir / "metrics_keras.json").open("w", encoding="utf-8") as handle:
            json.dump(result["metrics"], handle, indent=2)
        with (args.output_dir / "details_keras.json").open("w", encoding="utf-8") as handle:
            json.dump(result["details"], handle, indent=2)

    if args.mode in ("scratch", "both"):
        keras_model = tf.keras.models.load_model(args.model_path, compile=False)
        scratch_model = CaptionModel(decoder_type=args.decoder_type, injection_method=args.injection)
        scratch_model.load_from_keras(keras_model, word2idx, idx2word)
        result = evaluate(
            model=scratch_model,
            captions=captions,
            image_names=image_names,
            features=features,
            word2idx=word2idx,
            idx2word=idx2word,
            decoder_type=args.decoder_type,
            max_len=args.max_len,
            mode="scratch",
            seed=args.seed,
            decoding=args.decoding,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            injection_method=args.injection,
        )
        with (args.output_dir / "metrics_scratch.json").open("w", encoding="utf-8") as handle:
            json.dump(result["metrics"], handle, indent=2)
        with (args.output_dir / "details_scratch.json").open("w", encoding="utf-8") as handle:
            json.dump(result["details"], handle, indent=2)


if __name__ == "__main__":
    main()
