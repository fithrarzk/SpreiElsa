"""NumPy implementations of CNN forward layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import get_activation, get_activation_backward


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _ensure_batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    if x.ndim == 3:
        return x[np.newaxis, ...], True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Expected rank 3 or 4 input, got shape {x.shape}")


def _maybe_unbatch(x: np.ndarray, was_single: bool) -> np.ndarray:
    return x[0] if was_single else x


def _same_padding(input_size: int, kernel_size: int, stride: int) -> tuple[int, int]:
    output_size = int(np.ceil(input_size / stride))
    total_pad = max((output_size - 1) * stride + kernel_size - input_size, 0)
    before = total_pad // 2
    after = total_pad - before
    return before, after


def _pad_input(x: np.ndarray, kernel_size: tuple[int, int], strides: tuple[int, int], padding: str) -> np.ndarray:
    if padding == "valid":
        return x
    if padding != "same":
        raise ValueError(f"Unsupported padding: {padding}")
    pad_h = _same_padding(x.shape[1], kernel_size[0], strides[0])
    pad_w = _same_padding(x.shape[2], kernel_size[1], strides[1])
    return np.pad(x, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")


@dataclass
class Conv2D:
    kernel: np.ndarray
    bias: np.ndarray | None = None
    strides: tuple[int, int] = (1, 1)
    padding: str = "valid"
    activation: str | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        self._input = x
        self._was_single = was_single
        kh, kw, cin, cout = self.kernel.shape
        if x.shape[-1] != cin:
            raise ValueError(f"Expected {cin} input channels, got {x.shape[-1]}")

        strides = _pair(self.strides)
        padded = _pad_input(x, (kh, kw), strides, self.padding)
        self._padded = padded
        out_h = (padded.shape[1] - kh) // strides[0] + 1
        out_w = (padded.shape[2] - kw) // strides[1] + 1
        out = np.empty((padded.shape[0], out_h, out_w, cout), dtype=np.float32)

        for i in range(out_h):
            row = i * strides[0]
            for j in range(out_w):
                col = j * strides[1]
                patch = padded[:, row : row + kh, col : col + kw, :]
                out[:, i, j, :] = np.tensordot(patch, self.kernel, axes=([1, 2, 3], [0, 1, 2]))

        if self.bias is not None:
            out += self.bias.reshape((1, 1, 1, cout))
        self._pre_activation = out
        out = get_activation(self.activation)(out)
        self._activation_out = out
        return _maybe_unbatch(out, was_single)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        upstream, was_single = _ensure_batch(np.asarray(upstream, dtype=np.float32))
        if was_single != getattr(self, "_was_single", False):
            raise ValueError("Mismatch between forward and backward batch handling")

        dpre = get_activation_backward(self.activation)(self._pre_activation, self._activation_out, upstream)

        kh, kw, cin, cout = self.kernel.shape
        strides = _pair(self.strides)
        padded = self._padded
        out_h, out_w = dpre.shape[1], dpre.shape[2]
        grad_kernel = np.zeros_like(self.kernel)
        grad_input = np.zeros_like(padded)
        grad_bias = np.zeros((cout,), dtype=np.float32) if self.bias is not None else None

        for i in range(out_h):
            row = i * strides[0]
            for j in range(out_w):
                col = j * strides[1]
                patch = padded[:, row : row + kh, col : col + kw, :]
                dout_slice = dpre[:, i, j, :]
                grad_kernel += np.tensordot(patch, dout_slice, axes=([0], [0]))
                grad_input[:, row : row + kh, col : col + kw, :] += np.tensordot(dout_slice, self.kernel, axes=([1], [3]))

        if grad_bias is not None:
            grad_bias = np.sum(dpre, axis=(0, 1, 2))

        if self.padding == "same":
            pad_h = _same_padding(self._input.shape[1], kh, strides[0])
            pad_w = _same_padding(self._input.shape[2], kw, strides[1])
            grad_input = grad_input[:, pad_h[0] : grad_input.shape[1] - pad_h[1], pad_w[0] : grad_input.shape[2] - pad_w[1], :]

        self.grad_kernel = grad_kernel
        self.grad_bias = grad_bias
        return _maybe_unbatch(grad_input, was_single)


@dataclass
class LocallyConnected2D:
    kernel: np.ndarray
    bias: np.ndarray | None = None
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    padding: str = "valid"
    activation: str | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.padding != "valid":
            raise ValueError("LocallyConnected2D scratch supports valid padding only")
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        self._input = x
        self._was_single = was_single
        kh, kw = _pair(self.kernel_size)
        strides = _pair(self.strides)
        out_h = (x.shape[1] - kh) // strides[0] + 1
        out_w = (x.shape[2] - kw) // strides[1] + 1
        cout = self.kernel.shape[-1]
        out = np.empty((x.shape[0], out_h, out_w, cout), dtype=np.float32)

        position = 0
        for i in range(out_h):
            row = i * strides[0]
            for j in range(out_w):
                col = j * strides[1]
                patch = x[:, row : row + kh, col : col + kw, :].reshape((x.shape[0], -1))
                out[:, i, j, :] = patch @ self.kernel[position]
                position += 1

        if self.bias is not None:
            bias = self.bias.reshape((out_h * out_w, cout))
            out += bias.reshape((1, out_h, out_w, cout))
        self._pre_activation = out
        out = get_activation(self.activation)(out)
        self._activation_out = out
        return _maybe_unbatch(out, was_single)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        upstream, was_single = _ensure_batch(np.asarray(upstream, dtype=np.float32))
        if was_single != getattr(self, "_was_single", False):
            raise ValueError("Mismatch between forward and backward batch handling")

        dpre = get_activation_backward(self.activation)(self._pre_activation, self._activation_out, upstream)

        x = self._input
        kh, kw = _pair(self.kernel_size)
        strides = _pair(self.strides)
        out_h, out_w = dpre.shape[1], dpre.shape[2]
        cout = dpre.shape[-1]

        grad_input = np.zeros_like(x)
        grad_kernel = np.zeros_like(self.kernel)
        grad_bias = np.zeros((out_h * out_w, cout), dtype=np.float32) if self.bias is not None else None

        position = 0
        for i in range(out_h):
            row = i * strides[0]
            for j in range(out_w):
                col = j * strides[1]
                patch = x[:, row : row + kh, col : col + kw, :].reshape((x.shape[0], -1))
                dout_slice = dpre[:, i, j, :]
                grad_kernel[position] += patch.T @ dout_slice
                if grad_bias is not None:
                    grad_bias[position] += np.sum(dout_slice, axis=0)
                grad_patch = dout_slice @ self.kernel[position].T
                grad_input[:, row : row + kh, col : col + kw, :] += grad_patch.reshape((x.shape[0], kh, kw, -1))
                position += 1

        self.grad_kernel = grad_kernel
        self.grad_bias = grad_bias
        return _maybe_unbatch(grad_input, was_single)


@dataclass
class Pooling2D:
    pool_size: tuple[int, int] = (2, 2)
    strides: tuple[int, int] | None = None
    padding: str = "valid"
    mode: str = "max"

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.padding != "valid":
            raise ValueError("Pooling scratch supports valid padding only")
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        self._input = x
        self._was_single = was_single
        ph, pw = _pair(self.pool_size)
        strides = _pair(self.strides or self.pool_size)
        out_h = (x.shape[1] - ph) // strides[0] + 1
        out_w = (x.shape[2] - pw) // strides[1] + 1
        out = np.empty((x.shape[0], out_h, out_w, x.shape[-1]), dtype=np.float32)

        for i in range(out_h):
            row = i * strides[0]
            for j in range(out_w):
                col = j * strides[1]
                patch = x[:, row : row + ph, col : col + pw, :]
                if self.mode == "max":
                    out[:, i, j, :] = np.max(patch, axis=(1, 2))
                elif self.mode == "avg":
                    out[:, i, j, :] = np.mean(patch, axis=(1, 2))
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.mode}")
        return _maybe_unbatch(out, was_single)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        upstream, was_single = _ensure_batch(np.asarray(upstream, dtype=np.float32))
        if was_single != getattr(self, "_was_single", False):
            raise ValueError("Mismatch between forward and backward batch handling")

        x = self._input
        ph, pw = _pair(self.pool_size)
        strides = _pair(self.strides or self.pool_size)
        out_h, out_w = upstream.shape[1], upstream.shape[2]
        grad_input = np.zeros_like(x)

        for i in range(out_h):
            row = i * strides[0]
            for j in range(out_w):
                col = j * strides[1]
                patch = x[:, row : row + ph, col : col + pw, :]
                if self.mode == "avg":
                    grad = upstream[:, i, j, :][:, np.newaxis, np.newaxis, :] / float(ph * pw)
                    grad_input[:, row : row + ph, col : col + pw, :] += grad
                elif self.mode == "max":
                    max_vals = np.max(patch, axis=(1, 2), keepdims=True)
                    mask = patch == max_vals
                    counts = np.sum(mask, axis=(1, 2), keepdims=True)
                    grad = upstream[:, i, j, :][:, np.newaxis, np.newaxis, :]
                    grad_input[:, row : row + ph, col : col + pw, :] += mask * grad / np.maximum(counts, 1)
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.mode}")

        return _maybe_unbatch(grad_input, was_single)


class MaxPooling2D(Pooling2D):
    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] | None = None, padding: str = "valid"):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, mode="max")


class AveragePooling2D(Pooling2D):
    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] | None = None, padding: str = "valid"):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, mode="avg")


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        self._input = x
        self._was_single = was_single
        out = np.mean(x, axis=(1, 2))
        return _maybe_unbatch(out, was_single)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        upstream, was_single = _ensure_batch(np.asarray(upstream, dtype=np.float32))
        if was_single != getattr(self, "_was_single", False):
            raise ValueError("Mismatch between forward and backward batch handling")
        batch, height, width, channels = self._input.shape
        grad = upstream[:, np.newaxis, np.newaxis, :] / float(height * width)
        grad_input = np.ones_like(self._input) * grad
        return _maybe_unbatch(grad_input, was_single)


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        self._input = x
        self._was_single = was_single
        out = np.max(x, axis=(1, 2))
        return _maybe_unbatch(out, was_single)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        upstream, was_single = _ensure_batch(np.asarray(upstream, dtype=np.float32))
        if was_single != getattr(self, "_was_single", False):
            raise ValueError("Mismatch between forward and backward batch handling")
        x = self._input
        max_vals = np.max(x, axis=(1, 2), keepdims=True)
        mask = x == max_vals
        counts = np.sum(mask, axis=(1, 2), keepdims=True)
        grad = upstream[:, np.newaxis, np.newaxis, :]
        grad_input = mask * grad / np.maximum(counts, 1)
        return _maybe_unbatch(grad_input, was_single)


class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        self._input_shape = array.shape
        if array.ndim == 3:
            return array.reshape((-1,), order="C")
        if array.ndim == 4:
            return array.reshape((array.shape[0], -1), order="C")
        raise ValueError(f"Expected rank 3 or 4 input, got shape {array.shape}")

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_input_shape"):
            raise ValueError("Flatten backward called before forward")
        return np.asarray(upstream, dtype=np.float32).reshape(self._input_shape, order="C")


@dataclass
class Dense:
    kernel: np.ndarray
    bias: np.ndarray | None = None
    activation: str | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = np.asarray(x, dtype=np.float32)
        out = self._input @ self.kernel
        if self.bias is not None:
            out += self.bias
        self._pre_activation = out
        out = get_activation(self.activation)(out)
        self._activation_out = out
        return out

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        upstream = np.asarray(upstream, dtype=np.float32)
        dpre = get_activation_backward(self.activation)(self._pre_activation, self._activation_out, upstream)
        self.grad_kernel = self._input.T @ dpre
        self.grad_bias = np.sum(dpre, axis=0) if self.bias is not None else None
        return dpre @ self.kernel.T

