from typing import Dict, Optional
from pydantic import BaseModel, Field
import time

class ComputeCredit(BaseModel):
    """
    Represents a unit of Universal Basic Compute.
    1 Credit = 1 H100-Hour equivalent.
    """
    amount: float = Field(..., description="Amount of compute credits")
    source: str = Field(..., description="Source of the credit (e.g., 'UBI_GRANT', 'SOLAR_DIVIDEND')")
    timestamp: float = Field(default_factory=time.time, description="Time of issuance")
    energy_cost_joules: float = Field(..., description="Energy cost to produce this compute")

    def __repr__(self):
        return f"<ComputeCredit: {self.amount} H100-hrs from {self.source}>"

class ComputeWallet(BaseModel):
    """
    A citizen's wallet for storing and transacting Compute Credits.
    """
    owner_id: str
    balance: float = 0.0
    transactions: list[ComputeCredit] = []

    def deposit(self, amount: float, source: str, energy_cost: float):
        credit = ComputeCredit(amount=amount, source=source, energy_cost_joules=energy_cost)
        self.transactions.append(credit)
        self.balance += amount

    def spend(self, amount: float, purpose: str) -> bool:
        if self.balance >= amount:
            self.balance -= amount
            # Log spend (simplified)
            return True
        return False

class EnergyMarket(BaseModel):
    """
    Converts Energy (Joules) into Compute (FLOPS/H100-hrs).
    Simulates the 'Thermodynamic Peg'.
    """
    joules_per_h100_hour: float = 700.0 * 3600.0  # ~700W * 3600s
    grid_efficiency: float = 0.95

    def generate_compute(self, joules_input: float) -> float:
        """Returns H100-hours generated from Joules."""
        effective_energy = joules_input * self.grid_efficiency
        compute_generated = effective_energy / self.joules_per_h100_hour
        return compute_generated
