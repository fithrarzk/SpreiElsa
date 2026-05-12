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

    def forward_with_cache(self, x, h_prev):
        pre = x @ self.W_x + h_prev @ self.W_h + self.b
        h = np.tanh(pre)
        cache = (x, h_prev, pre, h)
        return h, cache

    def backward(self, upstream, cache):
        x, h_prev, pre, h = cache
        dtanh = upstream * (1.0 - h**2)
        grad_W_x = x.T @ dtanh
        grad_W_h = h_prev.T @ dtanh
        grad_b = np.sum(dtanh, axis=0)
        dx = dtanh @ self.W_x.T
        dh_prev = dtanh @ self.W_h.T
        return dx, dh_prev, grad_W_x, grad_W_h, grad_b

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

    def forward_with_cache(self, sequence, h0=None):
        batched = sequence.ndim == 3
        if not batched:
            sequence = sequence[np.newaxis]
        batch, seq_len, _ = sequence.shape
        h = h0 if h0 else [np.zeros((batch, self.cells[0].hidden_size), dtype=np.float32) for _ in self.cells]

        outputs = []
        caches = []
        for t in range(seq_len):
            x = sequence[:, t, :]
            step_caches = []
            for i, cell in enumerate(self.cells):
                h[i], cache = cell.forward_with_cache(x, h[i])
                step_caches.append(cache)
                x = h[i]
            outputs.append(x)
            caches.append(step_caches)

        out = np.stack(outputs, axis=1)
        if not batched:
            return out[0], [hi[0] for hi in h], caches
        return out, h, caches

    def backward(self, upstream, caches):
        batched = upstream.ndim == 3
        if not batched:
            upstream = upstream[np.newaxis]
        batch, seq_len, hidden = upstream.shape
        n_layers = len(self.cells)

        grad_W_x = [np.zeros_like(cell.W_x) for cell in self.cells]
        grad_W_h = [np.zeros_like(cell.W_h) for cell in self.cells]
        grad_b = [np.zeros_like(cell.b) for cell in self.cells]
        dh_next = [np.zeros((batch, hidden), dtype=np.float32) for _ in self.cells]

        grad_input = np.zeros((batch, seq_len, caches[0][0][0].shape[-1]), dtype=np.float32)

        for t in reversed(range(seq_len)):
            grad = upstream[:, t, :]
            for layer_idx in reversed(range(n_layers)):
                grad = grad + dh_next[layer_idx]
                cache = caches[t][layer_idx]
                dx, dh_prev, gWx, gWh, gb = self.cells[layer_idx].backward(grad, cache)
                grad_W_x[layer_idx] += gWx
                grad_W_h[layer_idx] += gWh
                grad_b[layer_idx] += gb
                dh_next[layer_idx] = dh_prev
                grad = dx
            grad_input[:, t, :] = grad

        grads = {"W_x": grad_W_x, "W_h": grad_W_h, "b": grad_b}
        if not batched:
            return grad_input[0], grads
        return grad_input, grads

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
