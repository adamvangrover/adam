import math
import random
from typing import Dict, Any, List

class PricingEngine:
    """
    Institutional-grade pricing engine.
    Handles multi-asset live pricing, spreads, and market making simulation using Geometric Brownian Motion (GBM).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mu = 0.1  # Drift
        self.sigma = 0.2  # Volatility
        self.dt = 1/252  # Daily time step

    def get_price(self, symbol: str, side: str = "mid", size: float = 1.0) -> Dict[str, float]:
        """
        Get the price for a specific asset using GBM simulation.
        """
        # Seed based on symbol hash to be consistent per-symbol momentarily,
        # but add time randomness in production. Here we want realistic volatility.
        # We start from a base price derived from symbol hash.
        seed_val = sum(ord(c) for c in symbol)
        base_price = (seed_val % 100) + 50.0

        # Simulate a random walk step from the base
        shock = random.gauss(0, 1)
        price_change = base_price * math.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * math.sqrt(self.dt) * shock)

        # Apply market impact based on size (Square root law approximation)
        market_impact = 0.1 * math.sqrt(size / 10000) if size > 0 else 0

        # Calculate Spread
        spread = self.simulate_spread(symbol, self.sigma)

        mid_price = price_change

        if side == "buy":
            # Buying pushes price up + half spread
            final_price = mid_price + (spread / 2) + market_impact
        elif side == "sell":
            # Selling pushes price down - half spread
            final_price = mid_price - (spread / 2) - market_impact
        else:
            final_price = mid_price

        return {
            "symbol": symbol,
            "side": side,
            "price": round(final_price, 4),
            "spread": round(spread, 4),
            "confidence": 0.95 + (0.04 * random.random()), # 0.95-0.99
            "source": "PricingEngine_GBM_v1"
        }

    def simulate_spread(self, symbol: str, volatility: float) -> float:
        """Simulates bid-ask spread based on volatility and liquidity regimes."""
        base_spread = 0.01
        # Higher volatility -> wider spread
        vol_component = volatility * 0.1
        # Random liquidity noise
        noise = random.uniform(0, 0.02)
        return base_spread + vol_component + noise
