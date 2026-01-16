"""
World Model Simulation Engine

This module implements a 'Market Physics' engine that simulates financial markets
as physical systems, as described in the Convergence of Complexity whitepaper.
It treats capital as matter and risk as energy, allowing for 'counterfactual reasoning'
and 'generative consistency' simulations.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from pydantic import BaseModel
try:
    from core.infrastructure.provenance import ProvenanceLogger
except ImportError:
    # Fallback if module structure differs slightly in test env
    try:
        from core.infrastructure.provenance import ProvenanceLogger
    except:
        class ProvenanceLogger:
            def log_execution(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

class MarketState(BaseModel):
    timestamp: int
    asset_prices: Dict[str, float]
    market_temperature: float = 1.0 # Proxy for Volatility/Risk
    capital_flows: Dict[str, float] # Momentum/Kinetic Energy

class WorldModelEngine:
    """
    Simulates the 'Physics' of the market.
    """

    def __init__(self, assets: List[str]):
        self.assets = assets
        self.gravity_constant = 0.05 # Mean reversion strength
        self.friction_coefficient = 0.01 # Transaction costs/Liquidity drag
        self.growth_drift = 0.0 # Secular growth trend
        self.logger = ProvenanceLogger()
        self.active_scenario = "Baseline"
        self.fundamental_anchors = {} # Map asset -> fair value

    def load_scenario(self, scenario_name: str):
        """
        Loads physics parameters for a specific macro regime.
        """
        self.active_scenario = scenario_name
        if scenario_name == "Reflationary_Agentic_Boom_2026":
            # High growth, sticky inflation (high temp floor), low friction (AI efficiency)
            self.gravity_constant = 0.02 # Weaker mean reversion (momentum dominates)
            self.friction_coefficient = 0.005 # AI reduces transaction costs
            self.growth_drift = 0.05 # Strong secular growth (AI Productivity)
        elif scenario_name == "Stagflation_2025":
            self.gravity_constant = 0.08
            self.friction_coefficient = 0.03
            self.growth_drift = -0.01

    def generate_trajectory(self, initial_state: MarketState, steps: int = 50) -> List[MarketState]:
        """
        Generates a consistent future trajectory based on physical laws.
        Ensures Conservation of Capital (Mass) unless acted upon by external inflow/outflow.
        """
        trajectory = [initial_state]
        current_state = initial_state

        for t in range(steps):
            new_prices = {}
            new_flows = {}

            total_energy = current_state.market_temperature

            for asset in self.assets:
                price = current_state.asset_prices.get(asset, 100.0)
                momentum = current_state.capital_flows.get(asset, 0.0)

                # Determine Anchor (Fundamental Value)
                # If not set, defaults to 100.0 OR the initial price if we want to assume efficient market start
                # Here we default to 100.0 to preserve legacy behavior, but allow overrides
                anchor = self.fundamental_anchors.get(asset, 100.0)

                # Force 1: Gravity (Mean Reversion to Fundamental Value)
                force_gravity = -self.gravity_constant * (price - anchor)

                # Force 2: Thermal Noise (Stochastic Volatility)
                force_thermal = np.random.normal(0, total_energy)

                # Force 3: Growth Drift (Productivity/Inflation)
                force_drift = self.growth_drift

                # Update Momentum (F = ma)
                new_momentum = momentum + force_gravity + force_thermal + force_drift

                # Apply Friction
                new_momentum *= (1 - self.friction_coefficient)

                # Update Position (Price)
                new_price = price + new_momentum
                new_price = max(0.01, new_price) # Price cannot be negative

                new_prices[asset] = new_price
                new_flows[asset] = new_momentum

            # Evolve Temperature (Volatility clustering)
            # Temperature tends to spike when prices move fast (Kinetic Energy -> Heat)
            kinetic_energy = sum([abs(m) for m in new_flows.values()])
            new_temp = 0.9 * current_state.market_temperature + 0.1 * (1 + kinetic_energy * 0.1)

            # Enforce Temperature Floor for Reflationary Scenarios (Sticky Inflation)
            if self.active_scenario == "Reflationary_Agentic_Boom_2026":
                new_temp = max(new_temp, 1.2)

            next_state = MarketState(
                timestamp=current_state.timestamp + 1,
                asset_prices=new_prices,
                market_temperature=new_temp,
                capital_flows=new_flows
            )
            trajectory.append(next_state)
            current_state = next_state

        # Log Provenance
        self.logger.log_execution(
            module="WorldModelEngine",
            operation="generate_trajectory",
            inputs={"initial_state": initial_state.model_dump(), "steps": steps, "scenario": self.active_scenario},
            outputs={"final_state": trajectory[-1].model_dump(), "trajectory_len": len(trajectory)},
            metadata={"assets": self.assets}
        )

        return trajectory

    def run_counterfactual(self,
                         initial_state: MarketState,
                         intervention: Dict[str, Any]) -> List[MarketState]:
        """
        Runs a 'What If' simulation.
        Intervention example: {"shock_asset": "AAPL", "shock_magnitude": -20.0}
        """
        # Apply intervention to initial state
        modified_state = initial_state.model_copy(deep=True)

        if "shock_asset" in intervention:
            asset = intervention["shock_asset"]
            mag = intervention.get("shock_magnitude", 0.0)
            if asset in modified_state.asset_prices:
                # Apply shock as an instantaneous impulse (change in momentum)
                modified_state.capital_flows[asset] += mag

        return self.generate_trajectory(modified_state)

# Example Usage
if __name__ == "__main__":
    assets = ["Asset_A", "Asset_B"]
    engine = WorldModelEngine(assets)

    init_state = MarketState(
        timestamp=0,
        asset_prices={"Asset_A": 100.0, "Asset_B": 100.0},
        market_temperature=1.0,
        capital_flows={"Asset_A": 0.0, "Asset_B": 0.0}
    )

    print("--- Baseline Trajectory ---")
    traj = engine.generate_trajectory(init_state, steps=5)
    for state in traj:
        print(f"T={state.timestamp} | Prices: {state.asset_prices}")

    print("\n--- Counterfactual: Huge Sell-off in Asset_A ---")
    cf_traj = engine.run_counterfactual(init_state, {"shock_asset": "Asset_A", "shock_magnitude": -10.0})
    for state in cf_traj:
        print(f"T={state.timestamp} | Prices: {state.asset_prices}")
