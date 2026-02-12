import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

class MarketRegime(Enum):
    NORMAL = "normal"
    STRESS = "stress"
    CRASH = "crash"

class RiskScenario(BaseModel):
    scenario_id: str
    volatility_regime: str
    risk_factors: Dict[str, float]
    is_tail_event: bool

class GenerativeRiskEngine:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_batch(self, n: int = 100, regime_mix: Dict[str, float] = None) -> List[RiskScenario]:
        """
        Generates a batch of risk scenarios.
        regime_mix: Dict mapping regime names to probability weights.
        """
        if regime_mix is None:
            regime_mix = {
                MarketRegime.NORMAL.value: 0.7,
                MarketRegime.STRESS.value: 0.25,
                MarketRegime.CRASH.value: 0.05
            }

        scenarios = []

        # Define parameters for each regime (Mean vector and Covariance matrix)
        # Factors: [GDP Growth, Inflation, Unemployment]

        params = {
            MarketRegime.NORMAL: {
                "mean": [2.5, 2.0, 4.0],
                "cov": [[1.0, 0.2, -0.3], [0.2, 0.5, 0.1], [-0.3, 0.1, 0.5]]
            },
            MarketRegime.STRESS: {
                "mean": [0.5, 4.5, 5.5],
                "cov": [[2.0, -0.5, 0.4], [-0.5, 2.0, 0.3], [0.4, 0.3, 1.0]]
            },
            MarketRegime.CRASH: {
                "mean": [-3.0, 1.5, 8.5],
                "cov": [[4.0, 0.8, -0.8], [0.8, 3.0, 0.5], [-0.8, 0.5, 2.5]]
            }
        }

        for i in range(n):
            # Pick a regime
            regime_key = np.random.choice(list(regime_mix.keys()), p=list(regime_mix.values()))
            regime = MarketRegime(regime_key)

            p = params[regime]

            # Generate sample
            sample = np.random.multivariate_normal(p["mean"], p["cov"])

            # Simple tail event detection (e.g., if any value is > 2 std devs from "Normal" mean)
            # But let's just define tail event as being in Crash regime with extreme values
            is_tail = False
            if regime == MarketRegime.CRASH:
                if sample[0] < -4.0 or sample[2] > 10.0:
                    is_tail = True

            scenario = RiskScenario(
                scenario_id=f"scn_{i}_{np.random.randint(1000,9999)}",
                volatility_regime=regime.value,
                risk_factors={
                    "gdp_growth": round(sample[0], 2),
                    "inflation": round(sample[1], 2),
                    "unemployment": round(sample[2], 2)
                },
                is_tail_event=is_tail
            )
            scenarios.append(scenario)

        return scenarios
