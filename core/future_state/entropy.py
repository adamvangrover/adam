from typing import List
from pydantic import BaseModel, Field
import random

class InformationCapital(BaseModel):
    """
    Represents an asset's value based on its information complexity.
    """
    name: str
    bit_depth: float = Field(..., description="Kolmogorov complexity in bits")
    mass_kg: float

    @property
    def value_density(self) -> float:
        return self.bit_depth / self.mass_kg

class ThermodynamicSystem(BaseModel):
    """
    Global state of Entropy vs Negentropy (Wealth).
    """
    total_energy_joules: float
    entropy_level: float = 1.0 # 0.0 (Perfect Order) to 1.0 (Heat Death)
    organized_information_bits: float = 0.0

    def update_entropy(self, energy_expended: float, information_created: float):
        """
        Expending energy to create information reduces local entropy.
        """
        # Simplified physics: dS = dQ/T
        entropy_reduction = information_created * 0.0001
        heat_waste = energy_expended * 0.5 # 50% efficiency

        self.organized_information_bits += information_created
        self.entropy_level = max(0.0, self.entropy_level - entropy_reduction + (heat_waste * 0.00001))

class MaxwellDemon(BaseModel):
    """
    The AGI agent that optimizes resource sorting.
    """
    efficiency_rating: float = 0.85

    def sort_resources(self, system: ThermodynamicSystem, energy_input: float) -> float:
        """
        Consumes energy to create InformationCapital (Negentropy).
        Returns bits of information created.
        """
        bits_created = energy_input * self.efficiency_rating * 1000 # 1 J = 1000 bits (mock)
        system.update_entropy(energy_input, bits_created)
        return bits_created
