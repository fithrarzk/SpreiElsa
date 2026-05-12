"""NumPy implementations of CNN forward layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import get_activation


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
        kh, kw, cin, cout = self.kernel.shape
        if x.shape[-1] != cin:
            raise ValueError(f"Expected {cin} input channels, got {x.shape[-1]}")

        strides = _pair(self.strides)
        padded = _pad_input(x, (kh, kw), strides, self.padding)
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
        out = get_activation(self.activation)(out)
        return _maybe_unbatch(out, was_single)


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
        out = get_activation(self.activation)(out)
        return _maybe_unbatch(out, was_single)


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


class MaxPooling2D(Pooling2D):
    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] | None = None, padding: str = "valid"):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, mode="max")


class AveragePooling2D(Pooling2D):
    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] | None = None, padding: str = "valid"):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, mode="avg")


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        out = np.mean(x, axis=(1, 2))
        return _maybe_unbatch(out, was_single)


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x, was_single = _ensure_batch(np.asarray(x, dtype=np.float32))
        out = np.max(x, axis=(1, 2))
        return _maybe_unbatch(out, was_single)


class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 3:
            return array.reshape((-1,), order="C")
        if array.ndim == 4:
            return array.reshape((array.shape[0], -1), order="C")
        raise ValueError(f"Expected rank 3 or 4 input, got shape {array.shape}")


@dataclass
class Dense:
    kernel: np.ndarray
    bias: np.ndarray | None = None
    activation: str | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32) @ self.kernel
        if self.bias is not None:
            out += self.bias
        return get_activation(self.activation)(out)

