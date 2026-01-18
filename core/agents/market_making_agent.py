import logging
import numpy as np
from typing import Dict, Any, Tuple
from core.quantitative.pricing import AvellanedaStoikovModel

class MarketMakingAgent:
    """
    Agentic wrapper around the Avellaneda-Stoikov model.
    Implements a 'System Brain' component that dynamically adapts risk aversion (Gamma)
    based on market conditions (Volatility, Inventory Risk), simulating an RL policy.
    """

    def __init__(self, initial_gamma: float = 0.5, initial_sigma: float = 0.2, dt: float = 1/252):
        self.model = AvellanedaStoikovModel(gamma=initial_gamma, sigma=initial_sigma, T=dt, k=1.5)
        self.logger = logging.getLogger(__name__)
        self.current_gamma = initial_gamma

        # RL-like Policy Parameters
        self.inventory_risk_threshold = 2000 # Units
        self.volatility_threshold = 30.0     # VIX level

    def adapt_strategy(self, market_state: Dict[str, Any]):
        """
        Dynamically tunes the 'Gamma' (Risk Aversion) parameter based on observation.

        Observation Space:
        - vix: Market Volatility Index
        - inventory: Current net position
        - tick_volatility: Realized volatility in the last window
        """
        vix = market_state.get("vix", 20.0)
        inventory = market_state.get("inventory", 0.0)

        # Base Gamma
        new_gamma = 0.1

        # 1. Volatility Scaling
        # If VIX > 30 (Crisis), increase risk aversion linearly
        if vix > 30:
            new_gamma += (vix - 30) * 0.05

        # 2. Inventory Risk Scaling
        # If absolute inventory is high, become extremely risk averse to force mean reversion
        inv_ratio = abs(inventory) / self.inventory_risk_threshold
        if inv_ratio > 1.0:
            new_gamma += (inv_ratio - 1.0) * 0.5

        # Clamp Gamma
        new_gamma = min(max(new_gamma, 0.01), 10.0)

        # Apply Smooth Update (Exponential Moving Average) to prevent jitter
        alpha = 0.2
        self.current_gamma = (alpha * new_gamma) + ((1 - alpha) * self.current_gamma)

        # Update Internal Model
        self.model.update_parameters(risk_aversion=self.current_gamma)

        # Update Sigma if provided
        if "sigma" in market_state:
            self.model.update_parameters(volatility=market_state["sigma"])

        return self.current_gamma

    def get_quotes(self, mid_price: float, inventory: float, time_remaining: float) -> Tuple[float, float]:
        """
        Delegates to the underlying quantitative model.
        """
        return self.model.get_quotes(mid_price, inventory, time_remaining)
