from __future__ import annotations

import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    return x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def linear_backward(pre: np.ndarray, out: np.ndarray, upstream: np.ndarray) -> np.ndarray:
    return upstream


def relu_backward(pre: np.ndarray, out: np.ndarray, upstream: np.ndarray) -> np.ndarray:
    return upstream * (pre > 0)


def softmax_backward(pre: np.ndarray, out: np.ndarray, upstream: np.ndarray, axis: int = -1) -> np.ndarray:
    summed = np.sum(upstream * out, axis=axis, keepdims=True)
    return out * (upstream - summed)


def get_activation(name: str | None):
    if name is None or name == "linear":
        return linear
    if name == "relu":
        return relu
    if name == "softmax":
        return softmax
    raise ValueError(f"Unsupported activation: {name}")


def get_activation_backward(name: str | None):
    if name is None or name == "linear":
        return linear_backward
    if name == "relu":
        return relu_backward
    if name == "softmax":
        return softmax_backward
    raise ValueError(f"Unsupported activation: {name}")

