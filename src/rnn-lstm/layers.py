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
        self._last_input = None

    def load_weights(self, keras_layer):
        self.W = keras_layer.get_weights()[0].astype(np.float32)

    def set_weights(self, W: np.ndarray):
        self.W = np.array(W, dtype=np.float32)

    def forward(self, x):
        self._last_input = np.array(x, copy=False)
        return self.W[x]

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if self._last_input is None:
            raise ValueError("Embedding backward called before forward")
        grad_W = np.zeros_like(self.W)
        indices = self._last_input
        if indices.ndim == 0:
            grad_W[int(indices)] += upstream
            self.grad_W = grad_W
            return np.zeros((), dtype=np.float32)
        if indices.ndim == 1:
            for i, idx in enumerate(indices):
                grad_W[int(idx)] += upstream[i]
        else:
            flat_idx = indices.reshape(-1)
            flat_grad = upstream.reshape((-1, upstream.shape[-1]))
            for i, idx in enumerate(flat_idx):
                grad_W[int(idx)] += flat_grad[i]
        self.grad_W = grad_W
        return np.zeros_like(indices, dtype=np.float32)


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
        self._input = np.asarray(x, dtype=np.float32)
        out = self._input @ self.W + self.b
        self._pre_activation = out
        self._output = self._act(out)
        return self._output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_input"):
            raise ValueError("Dense backward called before forward")
        upstream = np.asarray(upstream, dtype=np.float32)
        if self._act == _ACTIVATIONS["relu"]:
            dpre = upstream * (self._pre_activation > 0)
        elif self._act == _ACTIVATIONS["tanh"]:
            dpre = upstream * (1.0 - np.tanh(self._pre_activation) ** 2)
        elif self._act == _ACTIVATIONS["sigmoid"]:
            sig = _sigmoid(self._pre_activation)
            dpre = upstream * sig * (1.0 - sig)
        elif self._act == _ACTIVATIONS["softmax"]:
            summed = np.sum(upstream * self._output, axis=-1, keepdims=True)
            dpre = self._output * (upstream - summed)
        else:
            dpre = upstream
        self.grad_W = self._input.T @ dpre
        self.grad_b = np.sum(dpre, axis=0)
        return dpre @ self.W.T
