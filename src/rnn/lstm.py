import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class LSTMCell:

    def __init__(self):
        self.W_x: np.ndarray = None  # (input_dim, 4*hidden_size)
        self.W_h: np.ndarray = None  # (hidden_size, 4*hidden_size)
        self.b:   np.ndarray = None  # (4*hidden_size,)
        self._hidden_size: int = None

    def load_weights(self, keras_layer) -> None:
        w = keras_layer.get_weights()
        self.W_x = w[0].copy().astype(np.float32)
        self.W_h = w[1].copy().astype(np.float32)
        bias = w[2]
        self.b = (bias[0] + bias[1] if bias.ndim == 2 else bias.copy()).astype(np.float32)
        self._hidden_size = self.W_h.shape[0]

    def set_weights(self, W_x: np.ndarray, W_h: np.ndarray, b: np.ndarray) -> None:
        self.W_x = np.array(W_x, dtype=np.float32)
        self.W_h = np.array(W_h, dtype=np.float32)
        bias = np.array(b, dtype=np.float32)
        self.b = (bias[0] + bias[1] if bias.ndim == 2 else bias)
        self._hidden_size = self.W_h.shape[0]

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ):
        z = x @ self.W_x + h_prev @ self.W_h + self.b  # (..., 4*hidden)
        h = self._hidden_size

        i = _sigmoid(z[..., :h])
        f = _sigmoid(z[..., h:2*h])
        g = np.tanh(z[..., 2*h:3*h])
        o = _sigmoid(z[..., 3*h:4*h])

        c_t = f * c_prev + i * g
        h_t = o * np.tanh(c_t)
        return h_t, c_t

    @property
    def hidden_size(self) -> int:
        return self._hidden_size


class LSTMDecoder:

    def __init__(self, cells: list):
        self.cells = cells
        self.n_layers = len(cells)

    @classmethod
    def from_keras(cls, keras_model, n_layers: int, name_prefix: str = "lstm"):
        cells = []
        for i in range(n_layers):
            layer = keras_model.get_layer(f"{name_prefix}_{i}")
            cell = LSTMCell()
            cell.load_weights(layer)
            cells.append(cell)
        return cls(cells)

    def forward(
        self,
        sequence: np.ndarray,
        h0: list = None,
        c0: list = None,
    ):
        batched = sequence.ndim == 3
        if not batched:
            sequence = sequence[np.newaxis]

        batch, seq_len, _ = sequence.shape
        hidden_size = self.cells[0].hidden_size

        h = (
            [np.zeros((batch, hidden_size), dtype=np.float32) for _ in self.cells]
            if h0 is None
            else [np.array(s, dtype=np.float32) for s in h0]
        )
        c = (
            [np.zeros((batch, hidden_size), dtype=np.float32) for _ in self.cells]
            if c0 is None
            else [np.array(s, dtype=np.float32) for s in c0]
        )

        all_outputs = []
        for t in range(seq_len):
            x = sequence[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(x, h[layer_idx], c[layer_idx])
                x = h[layer_idx]
            all_outputs.append(x)

        outputs = np.stack(all_outputs, axis=1)  # (batch, seq_len, hidden_size)

        if not batched:
            outputs = outputs[0]
            h = [hi[0] for hi in h]
            c = [ci[0] for ci in c]

        return outputs, h, c

    def step(self, x: np.ndarray, h: list, c: list):
        batched = x.ndim == 2
        if not batched:
            x = x[np.newaxis]
            h = [hi[np.newaxis] for hi in h]
            c = [ci[np.newaxis] for ci in c]

        new_h, new_c = [], []
        inp = x
        for layer_idx, cell in enumerate(self.cells):
            h_new, c_new = cell.forward(inp, h[layer_idx], c[layer_idx])
            new_h.append(h_new)
            new_c.append(c_new)
            inp = h_new

        out = inp

        if not batched:
            out   = out[0]
            new_h = [hi[0] for hi in new_h]
            new_c = [ci[0] for ci in new_c]

        return out, new_h, new_c
