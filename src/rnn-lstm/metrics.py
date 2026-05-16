from __future__ import annotations

from collections import Counter
import math

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from text_utils import clean_caption


def tokenize(text: str) -> list[str]:
    cleaned = clean_caption(text)
    return cleaned.split() if cleaned else []


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(candidate: list[str], references: list[list[str]], n: int) -> float:
    if len(candidate) < n:
        return 0.0
    cand_counts = _ngram_counts(candidate, n)
    if not cand_counts:
        return 0.0
    max_counts: Counter = Counter()
    for reference in references:
        ref_counts = _ngram_counts(reference, n)
        for ngram, count in ref_counts.items():
            max_counts[ngram] = max(max_counts.get(ngram, 0), count)
    clipped = {ngram: min(count, max_counts.get(ngram, 0)) for ngram, count in cand_counts.items()}
    return sum(clipped.values()) / sum(cand_counts.values())


def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    if not candidate:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        p_n = _modified_precision(candidate, references, n)
        precisions.append(p_n if p_n > 0 else 1e-9)

    log_precision = sum((1.0 / max_n) * (0.0 if p == 0 else math.log(p)) for p in precisions)
    geo_mean = math.exp(log_precision)

    cand_len = len(candidate)
    ref_lens = [len(ref) for ref in references if ref]
    if not ref_lens:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda rlen: (abs(rlen - cand_len), rlen))

    if cand_len > closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (closest_ref_len / max(cand_len, 1)))
    return float(brevity_penalty * geo_mean)


def _count_matches(candidate: list[str], reference: list[str]) -> tuple[int, int]:
    if not candidate or not reference:
        return 0, 0
    ref_positions = {}
    for idx, token in enumerate(reference):
        ref_positions.setdefault(token, []).append(idx)

    matches = []
    last_pos = -1
    for token in candidate:
        positions = ref_positions.get(token)
        if not positions:
            continue
        next_pos = None
        for pos in positions:
            if pos > last_pos:
                next_pos = pos
                break
        if next_pos is None:
            continue
        matches.append(next_pos)
        last_pos = next_pos

    if not matches:
        return 0, 0

    chunks = 1
    for i in range(1, len(matches)):
        if matches[i] != matches[i - 1] + 1:
            chunks += 1
    return len(matches), chunks


def meteor_score(candidate: list[str], references: list[list[str]]) -> float:
    if not candidate:
        return 0.0
    best = 0.0
    for reference in references:
        if not reference:
            continue
        matches, chunks = _count_matches(candidate, reference)
        if matches == 0:
            continue
        precision = matches / len(candidate)
        recall = matches / len(reference)
        f_mean = (10 * precision * recall) / (recall + 9 * precision) if (precision + recall) > 0 else 0.0
        penalty = 0.5 * ((chunks / matches) ** 3) if matches else 0.0
        score = f_mean * (1.0 - penalty)
        if score > best:
            best = score
    return float(best)


