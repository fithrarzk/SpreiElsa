from __future__ import annotations

import numpy as np


def sequence_cross_entropy_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    pad_idx: int = 0,
    eps: float = 1e-9,
) -> tuple[float, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)

    if probs.ndim != 3:
        raise ValueError("probs must have shape (batch, seq_len, vocab)")
    if targets.shape != probs.shape[:2]:
        raise ValueError("targets must match probs batch/seq dimensions")

    batch, seq_len, vocab = probs.shape
    mask = targets != pad_idx
    valid = np.maximum(np.sum(mask), 1)

    idx_b = np.repeat(np.arange(batch), seq_len)
    idx_t = np.tile(np.arange(seq_len), batch)
    idx_v = targets.reshape(-1)

    flat_probs = probs.reshape((batch * seq_len, vocab))
    flat_mask = mask.reshape(-1)

    picked = flat_probs[np.arange(batch * seq_len), idx_v]
    picked = np.where(flat_mask, picked, 1.0)

    loss = -np.log(picked + eps)
    loss = float(np.sum(loss) / valid)

    grad = np.zeros_like(flat_probs)
    grad[np.arange(batch * seq_len), idx_v] = -1.0 / np.maximum(picked, eps)
    grad = grad.reshape((batch, seq_len, vocab))
    grad *= mask[:, :, np.newaxis]
    grad /= valid

    return loss, grad
