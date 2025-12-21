"""
Equation of the Infinite: A Unified Field Theory of Complexity
Implements the mathematical framework for calculating Systemic Complexity (C_omega).
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class ComplexityComponents:
    generative_capability: float  # A_gen
    quantum_speedup: float        # Q_speedup
    narrative_depth: float        # N_depth
    systemic_fragility: float     # R_fragility
    energy_cost: float           # E_energy


class ComplexityFieldTheory:
    """
    Implements the Unified Formula of Complexity:
    C_total = integral [ (A_gen * Q_speedup * N_depth) / (R_fragility * E_energy) ] dt
    """

    def __init__(self, time_steps: int = 100):
        self.time_steps = time_steps
        self.dt = 1.0 / time_steps

    def calculate_instantaneous_complexity(self, components: ComplexityComponents) -> float:
        """
        Calculates the instantaneous complexity at a single point in time.

        Formula: C = (A_gen * Q_speedup * N_depth) / (R_fragility * E_energy)
        """
        numerator = (
            components.generative_capability *
            components.quantum_speedup *
            components.narrative_depth
        )

        denominator = (
            components.systemic_fragility *
            components.energy_cost
        )

        # Prevent division by zero
        if denominator == 0:
            return float('inf')

        return numerator / denominator

    def simulate_trajectory(self,
                            initial_state: ComplexityComponents,
                            scenarios: Dict[str, float]) -> List[float]:
        """
        Simulates the complexity trajectory over time given growth/decay scenarios for each component.
        """
        complexity_trajectory = []
        current_state = initial_state

        for t in range(self.time_steps):
            c_t = self.calculate_instantaneous_complexity(current_state)
            complexity_trajectory.append(c_t)

            # Evolve state based on scenarios (simple linear growth for demo)
            current_state = ComplexityComponents(
                generative_capability=current_state.generative_capability * (1 + scenarios.get('A_gen_growth', 0)),
                quantum_speedup=current_state.quantum_speedup * (1 + scenarios.get('Q_speedup_growth', 0)),
                narrative_depth=current_state.narrative_depth * (1 + scenarios.get('N_depth_growth', 0)),
                systemic_fragility=current_state.systemic_fragility * (1 + scenarios.get('R_fragility_growth', 0)),
                energy_cost=current_state.energy_cost * (1 + scenarios.get('E_energy_growth', 0))
            )

        return complexity_trajectory

    def integrate_complexity(self, trajectory: List[float]) -> float:
        """
        Integrates the complexity trajectory over time using the Trapezoidal Rule.
        """
        integral = 0.0
        for i in range(len(trajectory) - 1):
            integral += 0.5 * (trajectory[i] + trajectory[i+1]) * self.dt
        return integral


def run_simulation():
    """
    Runs a sample simulation of the Complexity Field Theory.
    """
    print("Initializing Unified Field Theory Simulation...")

    # Baseline Components (Normalized)
    # A_gen: GAN/MoE capability (High)
    # Q_speedup: Quantum Advantage (Emerging, sqrt(N) for Amplitude Estimation)
    # N_depth: World Model dimensionality (Moving from 1D to 4D)
    # R_fragility: Inverse DSCR (High fragility means low DSCR, so high R)
    # E_energy: Joules per inference (High cost)

    initial_components = ComplexityComponents(
        generative_capability=100.0,  # High generative power
        quantum_speedup=10.0,         # ~sqrt(100) speedup
        narrative_depth=3.0,          # 3D World Models
        systemic_fragility=1.5,       # Moderate to High fragility (DSCR < 1.0 implies risk)
        energy_cost=50.0              # High energy cost
    )

    # Scenario: The "Stargate" Acceleration
    # Generative capability explodes, but Energy cost also rises.
    # Quantum speedup increases slowly.
    # Fragility increases due to leverage.
    scenarios = {
        'A_gen_growth': 0.05,       # 5% growth per step
        'Q_speedup_growth': 0.01,   # 1% growth per step
        'N_depth_growth': 0.02,     # 2% growth per step
        'R_fragility_growth': 0.01,  # 1% increase in fragility
        'E_energy_growth': 0.04     # 4% increase in energy cost
    }

    theory = ComplexityFieldTheory(time_steps=50)
    trajectory = theory.simulate_trajectory(initial_components, scenarios)
    total_complexity = theory.integrate_complexity(trajectory)

    print("\n--- Simulation Results ---")
    print(f"Initial Instantaneous Complexity: {trajectory[0]:.4f}")
    print(f"Final Instantaneous Complexity:   {trajectory[-1]:.4f}")
    print(f"Total Integrated Complexity (C_total): {total_complexity:.4f}")

    print("\n--- Insight ---")
    if trajectory[-1] > trajectory[0]:
        print("System is evolving towards higher complexity (Singularity approaching).")
        print("Note: Numerator growth (A_gen * Q * N) is outpacing Denominator drag (Fragility * Energy).")
    else:
        print("System complexity is collapsing due to thermodynamic or fragility constraints.")


if __name__ == "__main__":
    run_simulation()
