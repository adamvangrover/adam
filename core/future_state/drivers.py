from abc import ABC, abstractmethod
from pydantic import BaseModel
import math

class GrowthDriver(BaseModel, ABC):
    """
    Abstract base class for simulation drivers.
    """
    base_value: float
    rate: float

    @abstractmethod
    def value_at(self, time_step: float) -> float:
        pass

class ExponentialDriver(GrowthDriver):
    """
    Models exponential growth (e.g., Moore's Law).
    value = base * (1 + rate) ^ time_step
    """
    def value_at(self, time_step: float) -> float:
        return self.base_value * ((1.0 + self.rate) ** time_step)

class DecayDriver(GrowthDriver):
    """
    Models exponential decay (e.g., Cost of Compute).
    value = base * (1 - rate) ^ time_step
    """
    def value_at(self, time_step: float) -> float:
        return self.base_value * ((1.0 - self.rate) ** time_step)

class SigmoidDriver(GrowthDriver):
    """
    Models S-Curve adoption.
    value = capacity / (1 + e^(-k * (t - midpoint)))
    """
    capacity: float = 1.0
    midpoint: float = 10.0 # Time step where growth is steepest

    def value_at(self, time_step: float) -> float:
        k = self.rate
        try:
            val = self.capacity / (1.0 + math.exp(-k * (time_step - self.midpoint)))
        except OverflowError:
            val = self.capacity if time_step > self.midpoint else 0.0
        return val

class LinearDriver(GrowthDriver):
    """
    Models simple linear progression.
    """
    def value_at(self, time_step: float) -> float:
        return self.base_value + (self.rate * time_step)
