import numpy as np


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

_ACTIVATIONS = {
    "relu":    lambda x: np.maximum(0.0, x),
    "softmax": _softmax,
    "sigmoid": _sigmoid,
    "tanh":    np.tanh,
    None:      lambda x: x,
    "linear":  lambda x: x,
}


class EmbeddingLayer:
    def __init__(self):
        self.W = None  # (vocab_size, embed_dim)

    def load_weights(self, keras_layer):
        self.W = keras_layer.get_weights()[0].astype(np.float32)

    def set_weights(self, W: np.ndarray):
        self.W = np.array(W, dtype=np.float32)

    def forward(self, x):
        return self.W[x]


class DenseLayer:
    def __init__(self, activation=None):
        self.W = None  # (input_dim, output_dim)
        self.b = None  # (output_dim,)
        self._act = _ACTIVATIONS.get(activation, lambda x: x)

    def load_weights(self, keras_layer):
        w = keras_layer.get_weights()
        self.W = w[0].astype(np.float32)
        self.b = w[1].astype(np.float32)

    def set_weights(self, W: np.ndarray, b: np.ndarray):
        self.W = np.array(W, dtype=np.float32)
        self.b = np.array(b, dtype=np.float32)

    def forward(self, x):
        return self._act(x @ self.W + self.b)
