"""Sanity checks for scratch RNN/LSTM backward passes."""

from __future__ import annotations

import numpy as np

from rnn import SimpleRNNCell
from lstm import LSTMCell


def _relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(eps, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / denom))


def _rnn_loss(cell: SimpleRNNCell, x: np.ndarray, h_prev: np.ndarray) -> float:
    h, _ = cell.forward_with_cache(x, h_prev)
    return float(np.sum(h))


def check_rnn_cell_backward() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 3)).astype(np.float32)
    h_prev = rng.normal(size=(1, 2)).astype(np.float32)
    cell = SimpleRNNCell()
    cell.W_x = rng.normal(size=(3, 2)).astype(np.float32)
    cell.W_h = rng.normal(size=(2, 2)).astype(np.float32)
    cell.b = rng.normal(size=(2,)).astype(np.float32)

    h, cache = cell.forward_with_cache(x, h_prev)
    upstream = np.ones_like(h)
    _, _, gWx, gWh, gb = cell.backward(upstream, cache)

    eps = 1e-4
    grad_num_Wx = np.zeros_like(cell.W_x)
    for i in range(cell.W_x.shape[0]):
        for j in range(cell.W_x.shape[1]):
            orig = cell.W_x[i, j]
            cell.W_x[i, j] = orig + eps
            loss_pos = _rnn_loss(cell, x, h_prev)
            cell.W_x[i, j] = orig - eps
            loss_neg = _rnn_loss(cell, x, h_prev)
            grad_num_Wx[i, j] = (loss_pos - loss_neg) / (2 * eps)
            cell.W_x[i, j] = orig

    grad_num_Wy = np.zeros_like(cell.W_h)
    for i in range(cell.W_h.shape[0]):
        for j in range(cell.W_h.shape[1]):
            orig = cell.W_h[i, j]
            cell.W_h[i, j] = orig + eps
            loss_pos = _rnn_loss(cell, x, h_prev)
            cell.W_h[i, j] = orig - eps
            loss_neg = _rnn_loss(cell, x, h_prev)
            grad_num_Wy[i, j] = (loss_pos - loss_neg) / (2 * eps)
            cell.W_h[i, j] = orig

    grad_num_b = np.zeros_like(cell.b)
    for i in range(cell.b.shape[0]):
        orig = cell.b[i]
        cell.b[i] = orig + eps
        loss_pos = _rnn_loss(cell, x, h_prev)
        cell.b[i] = orig - eps
        loss_neg = _rnn_loss(cell, x, h_prev)
        grad_num_b[i] = (loss_pos - loss_neg) / (2 * eps)
        cell.b[i] = orig

    print("SimpleRNNCell grad Wx rel err:", _relative_error(gWx, grad_num_Wx))
    print("SimpleRNNCell grad Wh rel err:", _relative_error(gWh, grad_num_Wy))
    print("SimpleRNNCell grad b rel err:", _relative_error(gb, grad_num_b))


def _lstm_loss(cell: LSTMCell, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> float:
    h, _, _ = cell.forward_with_cache(x, h_prev, c_prev)
    return float(np.sum(h))


def check_lstm_cell_backward() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 3)).astype(np.float32)
    h_prev = rng.normal(size=(1, 2)).astype(np.float32)
    c_prev = rng.normal(size=(1, 2)).astype(np.float32)

    cell = LSTMCell()
    W_x = rng.normal(size=(3, 8)).astype(np.float32)
    W_h = rng.normal(size=(2, 8)).astype(np.float32)
    b = rng.normal(size=(8,)).astype(np.float32)
    cell.set_weights(W_x, W_h, b)

    h, _, cache = cell.forward_with_cache(x, h_prev, c_prev)
    upstream = np.ones_like(h)
    dc = np.zeros_like(h)
    _, _, _, gWx, gWh, gb = cell.backward(upstream, dc, cache)

    eps = 1e-4
    grad_num_Wx = np.zeros_like(cell.W_x)
    for i in range(cell.W_x.shape[0]):
        for j in range(cell.W_x.shape[1]):
            orig = cell.W_x[i, j]
            cell.W_x[i, j] = orig + eps
            loss_pos = _lstm_loss(cell, x, h_prev, c_prev)
            cell.W_x[i, j] = orig - eps
            loss_neg = _lstm_loss(cell, x, h_prev, c_prev)
            grad_num_Wx[i, j] = (loss_pos - loss_neg) / (2 * eps)
            cell.W_x[i, j] = orig

    grad_num_Wy = np.zeros_like(cell.W_h)
    for i in range(cell.W_h.shape[0]):
        for j in range(cell.W_h.shape[1]):
            orig = cell.W_h[i, j]
            cell.W_h[i, j] = orig + eps
            loss_pos = _lstm_loss(cell, x, h_prev, c_prev)
            cell.W_h[i, j] = orig - eps
            loss_neg = _lstm_loss(cell, x, h_prev, c_prev)
            grad_num_Wy[i, j] = (loss_pos - loss_neg) / (2 * eps)
            cell.W_h[i, j] = orig

    grad_num_b = np.zeros_like(cell.b)
    for i in range(cell.b.shape[0]):
        orig = cell.b[i]
        cell.b[i] = orig + eps
        loss_pos = _lstm_loss(cell, x, h_prev, c_prev)
        cell.b[i] = orig - eps
        loss_neg = _lstm_loss(cell, x, h_prev, c_prev)
        grad_num_b[i] = (loss_pos - loss_neg) / (2 * eps)
        cell.b[i] = orig

    print("LSTMCell grad Wx rel err:", _relative_error(gWx, grad_num_Wx))
    print("LSTMCell grad Wh rel err:", _relative_error(gWh, grad_num_Wy))
    print("LSTMCell grad b rel err:", _relative_error(gb, grad_num_b))


def main() -> None:
    check_rnn_cell_backward()
    check_lstm_cell_backward()


if __name__ == "__main__":
    main()
