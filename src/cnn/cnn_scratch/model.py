from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SequentialScratchModel:
    layers: list

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        grad = upstream
        for layer in reversed(self.layers):
            if not hasattr(layer, "backward"):
                raise ValueError(f"Layer {layer.__class__.__name__} does not implement backward")
            grad = layer.backward(grad)
        return grad


def from_layers(layers: Iterable) -> SequentialScratchModel:
    return SequentialScratchModel(list(layers))

