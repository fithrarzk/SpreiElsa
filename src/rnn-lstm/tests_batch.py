"""Sanity checks for batch inference in scratch RNN/LSTM."""

from __future__ import annotations

import numpy as np

from rnn import SimpleRNNCell, SimpleRNNDecoder
from lstm import LSTMCell, LSTMDecoder


def check_rnn_batch() -> None:
    rng = np.random.default_rng(0)
    batch, seq_len, input_dim, hidden = 2, 5, 4, 3
    x = rng.normal(size=(batch, seq_len, input_dim)).astype(np.float32)

    cell = SimpleRNNCell()
    cell.W_x = rng.normal(size=(input_dim, hidden)).astype(np.float32)
    cell.W_h = rng.normal(size=(hidden, hidden)).astype(np.float32)
    cell.b = rng.normal(size=(hidden,)).astype(np.float32)

    decoder = SimpleRNNDecoder([cell])
    outputs, h = decoder.forward(x)
    assert outputs.shape == (batch, seq_len, hidden), outputs.shape
    print("RNN batch output shape:", outputs.shape)


def check_lstm_batch() -> None:
    rng = np.random.default_rng(1)
    batch, seq_len, input_dim, hidden = 2, 6, 4, 3
    x = rng.normal(size=(batch, seq_len, input_dim)).astype(np.float32)

    cell = LSTMCell()
    W_x = rng.normal(size=(input_dim, 4 * hidden)).astype(np.float32)
    W_h = rng.normal(size=(hidden, 4 * hidden)).astype(np.float32)
    b = rng.normal(size=(4 * hidden,)).astype(np.float32)
    cell.set_weights(W_x, W_h, b)

    decoder = LSTMDecoder([cell])
    outputs, h, c = decoder.forward(x)
    assert outputs.shape == (batch, seq_len, hidden), outputs.shape
    print("LSTM batch output shape:", outputs.shape)


def main() -> None:
    check_rnn_batch()
    check_lstm_batch()


if __name__ == "__main__":
    main()
