"""Sanity checks for batch inference in scratch CNN."""

from __future__ import annotations

import numpy as np

from src.cnn.cnn_scratch.layers import Conv2D, Dense, Flatten, LocallyConnected2D, MaxPooling2D
from src.cnn.cnn_scratch.model import SequentialScratchModel


def check_shared_batch() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8, 8, 3)).astype(np.float32)
    kernel = rng.normal(size=(3, 3, 3, 4)).astype(np.float32)
    bias = np.zeros((4,), dtype=np.float32)

    conv = Conv2D(kernel=kernel, bias=bias, activation="relu")
    pool = MaxPooling2D(pool_size=(2, 2))
    flat = Flatten()
    dense = Dense(rng.normal(size=(3 * 3 * 4, 5)).astype(np.float32), np.zeros((5,), dtype=np.float32))

    model = SequentialScratchModel([conv, pool, flat, dense])
    out = model.predict(x)
    assert out.shape == (4, 5), out.shape
    print("Shared batch output shape:", out.shape)


def check_non_shared_batch() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 8, 8, 3)).astype(np.float32)
    kh, kw, cin, cout = 3, 3, 3, 2
    out_h = (8 - kh) + 1
    out_w = (8 - kw) + 1
    positions = out_h * out_w
    kernel = rng.normal(size=(positions, kh * kw * cin, cout)).astype(np.float32)
    bias = np.zeros((positions, cout), dtype=np.float32)

    local = LocallyConnected2D(kernel=kernel, bias=bias, kernel_size=(kh, kw), activation="relu")
    out = local.forward(x)
    assert out.shape == (3, out_h, out_w, cout), out.shape
    print("Non-shared batch output shape:", out.shape)


def main() -> None:
    check_shared_batch()
    check_non_shared_batch()


if __name__ == "__main__":
    main()
