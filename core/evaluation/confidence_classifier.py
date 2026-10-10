from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence

@dataclass(frozen=True)
class ConfidenceModel:
    weights: tuple[float, ...]
    bias: float
    version: str

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature dimension mismatch")

        z = self.bias + sum(
            weight * value
            for weight, value in zip(self.weights, features)
        )

        # Numerically stable sigmoid.
        if z >= 0:
            e = exp(-z)
            return 1.0 / (1.0 + e)

        e = exp(z)
        return e / (1.0 + e)
