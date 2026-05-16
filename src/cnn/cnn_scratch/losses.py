from __future__ import annotations

import numpy as np


def softmax_cross_entropy_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-9,
    reduction: str = "mean",
) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    single = logits.ndim == 1
    if single:
        logits = logits[np.newaxis, :]
        labels = labels.reshape((1,))

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    batch = logits.shape[0]
    idx = np.arange(batch)
    loss = -np.log(probs[idx, labels] + eps)
    if reduction == "mean":
        loss = float(np.mean(loss))
    elif reduction == "sum":
        loss = float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    grad = probs
    grad[idx, labels] -= 1.0
    if reduction == "mean":
        grad /= batch

    if single:
        grad = grad[0]

    return loss, grad


def cross_entropy_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-9,
    reduction: str = "mean",
) -> tuple[float, np.ndarray]:

    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    single = probs.ndim == 1
    if single:
        probs = probs[np.newaxis, :]
        labels = labels.reshape((1,))

    batch = probs.shape[0]
    idx = np.arange(batch)
    loss = -np.log(probs[idx, labels] + eps)
    if reduction == "mean":
        loss = float(np.mean(loss))
    elif reduction == "sum":
        loss = float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    grad = np.zeros_like(probs)
    grad[idx, labels] = -1.0 / np.maximum(probs[idx, labels], eps)
    if reduction == "mean":
        grad /= batch

    if single:
        grad = grad[0]

    return loss, grad
