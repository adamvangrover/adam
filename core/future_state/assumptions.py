from pydantic import BaseModel, Field

class SimulationAssumptions(BaseModel):
    """
    Central configuration for the Singularity Simulation.
    Defines the starting conditions and growth parameters.
    """

    # 1. Timeline
    START_YEAR: int = 2025
    END_YEAR: int = 2125

    # 2. Growth Constants
    MOORES_LAW_DOUBLING_MONTHS: float = Field(18.0, description="Months to double compute/cost")
    ENERGY_COST_DECAY_RATE: float = Field(0.05, description="Annual % decrease in energy cost")
    ALGO_EFFICIENCY_MULTIPLIER: float = Field(2.0, description="Software efficiency gains per decade (independent of hardware)")

    # 3. Thresholds
    SINGULARITY_COMPUTE_THRESHOLD_FLOPS: float = 1e24 # 1 Yottaflop for "Singularity"
    POST_SCARCITY_ENERGY_COST_THRESHOLD: float = 0.001 # $0.001 per kWh

    # 4. Economic Baselines (Trillions USD)
    INITIAL_HUMAN_GDP: float = 100.0
    INITIAL_AI_GDP: float = 0.1

    # 5. Volatility
    ECONOMIC_VOLATILITY_FACTOR: float = 0.1 # Random noise in GDP growth

    # 6. Alignment
    DEFAULT_ALIGNMENT_DRIFT: float = -0.01 # Entropy tends to degrade alignment without effort

    class Config:
        validate_assignment = True
