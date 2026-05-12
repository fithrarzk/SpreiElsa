"""Sequential scratch model utilities."""

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


def from_layers(layers: Iterable) -> SequentialScratchModel:
    return SequentialScratchModel(list(layers))

