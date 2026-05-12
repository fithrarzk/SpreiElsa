"""Sanity checks for scratch CNN backward passes."""

from __future__ import annotations

import numpy as np

from src.cnn.cnn_scratch.layers import Conv2D, Dense
from src.cnn.cnn_scratch.losses import softmax_cross_entropy_loss


def _relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(eps, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / denom))


def _dense_loss(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> float:
    out = x @ kernel + bias
    return float(np.sum(out))


def check_dense_backward() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4)).astype(np.float32)
    kernel = rng.normal(size=(4, 3)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)

    layer = Dense(kernel.copy(), bias.copy(), activation=None)
    out = layer.forward(x)
    upstream = np.ones_like(out)
    layer.backward(upstream)

    eps = 1e-4
    grad_num = np.zeros_like(kernel)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            orig = kernel[i, j]
            kernel[i, j] = orig + eps
            loss_pos = _dense_loss(x, kernel, bias)
            kernel[i, j] = orig - eps
            loss_neg = _dense_loss(x, kernel, bias)
            grad_num[i, j] = (loss_pos - loss_neg) / (2 * eps)
            kernel[i, j] = orig

    err = _relative_error(layer.grad_kernel, grad_num)
    print("Dense grad kernel rel err:", err)


def _conv_loss(x: np.ndarray, kernel: np.ndarray) -> float:
    layer = Conv2D(kernel=kernel, bias=None, strides=(1, 1), padding="valid", activation=None)
    out = layer.forward(x)
    return float(np.sum(out))


def check_conv2d_backward() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 4, 4, 1)).astype(np.float32)
    kernel = rng.normal(size=(3, 3, 1, 1)).astype(np.float32)

    layer = Conv2D(kernel=kernel.copy(), bias=None, strides=(1, 1), padding="valid", activation=None)
    out = layer.forward(x)
    upstream = np.ones_like(out)
    layer.backward(upstream)

    eps = 1e-4
    grad_num = np.zeros_like(kernel)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for c in range(kernel.shape[2]):
                for k in range(kernel.shape[3]):
                    orig = kernel[i, j, c, k]
                    kernel[i, j, c, k] = orig + eps
                    loss_pos = _conv_loss(x, kernel)
                    kernel[i, j, c, k] = orig - eps
                    loss_neg = _conv_loss(x, kernel)
                    grad_num[i, j, c, k] = (loss_pos - loss_neg) / (2 * eps)
                    kernel[i, j, c, k] = orig

    err = _relative_error(layer.grad_kernel, grad_num)
    print("Conv2D grad kernel rel err:", err)


def main() -> None:
    check_dense_backward()
    check_conv2d_backward()
    logits = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)
    loss, grad = softmax_cross_entropy_loss(logits, labels)
    print("Softmax CE loss:", loss, "grad shape:", grad.shape)


if __name__ == "__main__":
    main()
