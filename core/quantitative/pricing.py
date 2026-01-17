import math
import numpy as np
from typing import Tuple

class AvellanedaStoikovModel:
    """
    Implements the Avellaneda-Stoikov (2008) market making model.
    Calculates optimal reservation price and spread based on inventory risk.
    """

    def __init__(self, gamma: float = 0.1, sigma: float = 0.2, T: float = 1.0, k: float = 1.5):
        """
        Args:
            gamma: Risk aversion parameter.
            sigma: Market volatility (annualized).
            T: Trading session length (normalized, e.g., 1 day = 1/252).
            k: Order book density / liquidity parameter.
        """
        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        self.k = k

    def get_reservation_price(self, mid_price: float, inventory_q: float, time_remaining: float) -> float:
        """
        Calculates the reservation price r(s, t).
        r(s, t) = s - q * gamma * sigma^2 * (T - t)

        If inventory is positive (long), reservation price drops (eager to sell).
        If inventory is negative (short), reservation price rises (eager to buy).
        """
        # Ensure time remaining is non-negative
        t_rem = max(0.0, time_remaining)

        r = mid_price - (inventory_q * self.gamma * (self.sigma ** 2) * t_rem)
        return r

    def get_optimal_spread(self) -> float:
        """
        Calculates the optimal spread delta.
        delta = (2/gamma) * ln(1 + gamma/k)

        Note: This is the simplified approximation. The full model depends on time,
        but usually delta is constant-ish relative to inventory in the approximation.
        Strictly speaking: delta + (2/gamma) * ln(...)

        Let's use the formula from the blueprint:
        delta = (2 / gamma) * ln(1 + (gamma / k))
        """
        # Avoid division by zero if gamma is 0 (risk neutral)
        if self.gamma < 1e-6:
            return 0.0 # Risk neutral market maker just quotes spread ~ 0 or transaction cost

        spread = (2.0 / self.gamma) * math.log(1 + (self.gamma / self.k))
        return spread

    def get_quotes(self, mid_price: float, inventory_q: float, time_remaining: float) -> Tuple[float, float]:
        """
        Returns (Bid, Ask) prices.
        Bid = r - delta/2
        Ask = r + delta/2
        """
        r = self.get_reservation_price(mid_price, inventory_q, time_remaining)
        delta = self.get_optimal_spread()

        bid = r - (delta / 2.0)
        ask = r + (delta / 2.0)

        return (bid, ask)

    def update_parameters(self, volatility: float = None, risk_aversion: float = None):
        """
        Allows dynamic updating of parameters (e.g. from RL agent or Scenario).
        """
        if volatility is not None:
            self.sigma = volatility
        if risk_aversion is not None:
            self.gamma = risk_aversion
