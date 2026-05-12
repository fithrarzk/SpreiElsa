import numpy as np


class SimpleRNNCell:
    def __init__(self):
        self.W_x = None  # (input_dim, hidden_size)
        self.W_h = None  # (hidden_size, hidden_size)
        self.b   = None  # (hidden_size,)

    def load_weights(self, keras_layer):
        w = keras_layer.get_weights()
        self.W_x, self.W_h, self.b = w[0].astype(np.float32), w[1].astype(np.float32), w[2].astype(np.float32)

    def forward(self, x, h_prev):
        return np.tanh(x @ self.W_x + h_prev @ self.W_h + self.b)

    @property
    def hidden_size(self):
        return self.W_h.shape[0]


class SimpleRNNDecoder:
    def __init__(self, cells):
        self.cells = cells
        self.n_layers = len(cells)

    def forward(self, sequence, h0=None):
        batched = sequence.ndim == 3
        if not batched:
            sequence = sequence[np.newaxis]
        batch, seq_len, _ = sequence.shape
        h = h0 if h0 else [np.zeros((batch, self.cells[0].hidden_size), dtype=np.float32) for _ in self.cells]

        outputs = []
        for t in range(seq_len):
            x = sequence[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell.forward(x, h[i])
                x = h[i]
            outputs.append(x)

        out = np.stack(outputs, axis=1)
        if not batched:
            return out[0], [hi[0] for hi in h]
        return out, h

    def step(self, x, h):
        batched = x.ndim == 2
        if not batched:
            x, h = x[np.newaxis], [hi[np.newaxis] for hi in h]
        new_h = []
        for i, cell in enumerate(self.cells):
            h[i] = cell.forward(x, h[i])
            x = h[i]
            new_h.append(h[i])
        if not batched:
            return x[0], [hi[0] for hi in new_h]
        return x, new_h
