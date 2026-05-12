"""Lightweight sanity checks for scratch CNN components."""

from __future__ import annotations

import numpy as np

from src.cnn.cnn_scratch.activations import softmax
from src.cnn.cnn_scratch.layers import Conv2D, Dense, Flatten, MaxPooling2D


def main() -> None:
    x = np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape((1, 4, 4, 1))
    kernel = np.ones((3, 3, 1, 2), dtype=np.float32)
    bias = np.array([0.0, 1.0], dtype=np.float32)
    conv = Conv2D(kernel, bias=bias, activation="relu")
    conv_out = conv.forward(x)
    assert conv_out.shape == (1, 2, 2, 2), conv_out.shape

    pool = MaxPooling2D(pool_size=(2, 2))
    pool_out = pool.forward(x)
    assert pool_out.tolist() == [[[[5.0], [7.0]], [[13.0], [15.0]]]], pool_out

    flat = Flatten().forward(x)
    assert flat.shape == (1, 16)
    assert flat[0, 0] == 0 and flat[0, -1] == 15

    dense = Dense(np.ones((16, 3), dtype=np.float32), np.zeros((3,), dtype=np.float32), activation="softmax")
    probs = dense.forward(flat)
    assert probs.shape == (1, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert np.allclose(np.sum(softmax(np.array([[1.0, 2.0, 3.0]])), axis=1), 1.0)
    try:
        import tensorflow as tf

        keras_conv = tf.keras.layers.Conv2D(2, (3, 3), activation="relu")
        keras_conv.build((None, 5, 5, 1))
        keras_kernel = np.ones((3, 3, 1, 2), dtype=np.float32)
        keras_bias = np.array([0.0, 1.0], dtype=np.float32)
        keras_conv.set_weights([keras_kernel, keras_bias])
        keras_out = keras_conv(np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1) / 10.0).numpy()
        scratch_out = Conv2D(keras_kernel, keras_bias, activation="relu").forward(
            np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1) / 10.0
        )
        assert np.allclose(keras_out, scratch_out, atol=1e-5)
    except ImportError:
        print("tensorflow unavailable; skipped Conv2D Keras comparison")
    print("scratch sanity checks passed")


if __name__ == "__main__":
    main()
