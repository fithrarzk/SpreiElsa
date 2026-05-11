import re
import json
import numpy as np
from collections import Counter

PAD_IDX, START_IDX, END_IDX, UNK_IDX = 0, 1, 2, 3


def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_flickr8k_captions(captions_file):
    captions = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("image,caption"):
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            img_name = parts[0].split("#")[0].strip()
            captions.setdefault(img_name, []).append(parts[1].strip())
    return captions


def load_split_names(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_vocab(captions_list, min_freq=1):
    counter = Counter()
    for caption in captions_list:
        counter.update(clean_caption(caption).split())

    special = ["<pad>", "<start>", "<end>", "<unk>"]
    words = special + [w for w, freq in counter.most_common() if freq >= min_freq]
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def encode_caption(caption, word2idx, max_len=None):
    tokens = clean_caption(caption).split()
    encoded = [word2idx["<start>"]] + [word2idx.get(t, UNK_IDX) for t in tokens] + [word2idx["<end>"]]
    if max_len and len(encoded) > max_len:
        encoded = encoded[:max_len - 1] + [word2idx["<end>"]]
    return encoded


def decode_caption(indices, idx2word, stop_at_end=True):
    words = []
    for idx in indices:
        word = idx2word.get(int(idx), "<unk>")
        if stop_at_end and word == "<end>":
            break
        if word not in ("<start>", "<pad>"):
            words.append(word)
    return " ".join(words)


def prepare_training_pair(caption, word2idx, max_len=35):
    encoded = encode_caption(caption, word2idx, max_len=max_len)
    padded = (encoded + [PAD_IDX] * max_len)[:max_len]
    return np.array(padded[:-1], dtype=np.int32), np.array(padded[1:], dtype=np.int32)


def pad_sequences(sequences, max_len, pad_value=PAD_IDX):
    out = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        out[i, :min(len(seq), max_len)] = seq[:max_len]
    return out


def save_vocab(word2idx, idx2word, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "idx2word": {str(k): v for k, v in idx2word.items()}}, f, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["word2idx"], {int(k): v for k, v in data["idx2word"].items()}


def caption_length_stats(captions_list, word2idx):
    lengths = np.array([len(encode_caption(c, word2idx)) for c in captions_list])
    return {"min": int(lengths.min()), "max": int(lengths.max()),
            "mean": float(lengths.mean()), "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99))}
