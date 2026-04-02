import math
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Attempt to load the high-performance Rust bindings (Avellaneda-Stoikov model)
# Graceful degradation: Fall back to Python implementation if Rust module is missing or fails to load.
try:
    import rust_pricing
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("rust_pricing module not found. Falling back to pure Python implementation for AvellanedaStoikovModel. Run 'maturin develop' in 'core/rust_pricing/' to compile.")

class AvellanedaStoikovModel:
    """
    Implements the Avellaneda-Stoikov (2008) market making model.
    Calculates optimal reservation price and spread based on inventory risk.
    Utilizes Rust bindings for deterministic, sub-millisecond execution if available.
    """

    def __init__(self, gamma: float = 0.1, sigma: float = 0.2, T: float = 1.0, k: float = 1.5):
        """
        Args:
            gamma: Risk aversion parameter.
            sigma: Market volatility (annualized).
            T: Trading session length (normalized, e.g., 1 day = 1/252).
            k: Order book density / liquidity parameter (Kappa).
        """
        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        self.k = k

    def get_reservation_price(self, mid_price: float, inventory_q: float, time_remaining: float) -> float:
        """
        Calculates the reservation price r(s, t).
        r(s, t) = s - q * gamma * sigma^2 * (T - t)
        """
        t_rem = max(0.0, time_remaining)
        r = mid_price - (inventory_q * self.gamma * (self.sigma ** 2) * t_rem)
        return r

    def get_optimal_spread(self, time_remaining: float = 1.0) -> float:
        """
        Calculates the optimal spread delta based on the exact formula:
        delta = (2/gamma) * ln(1 + gamma/k) + gamma * sigma^2 * (T-t)

        Note: The pure python version uses the simplified approximation by default
        in some contexts, but here we align with the exact Rust calculation.
        """
        if self.gamma < 1e-6:
            return 0.0

        t_rem = max(0.0, time_remaining)
        spread_term_1 = (2.0 / self.gamma) * math.log(1 + (self.gamma / self.k))
        spread_term_2 = self.gamma * (self.sigma ** 2) * t_rem

        return spread_term_1 + spread_term_2

    def get_quotes(self, mid_price: float, inventory_q: float, time_remaining: float) -> Tuple[float, float]:
        """
        Returns (Bid, Ask) prices.
        If RUST_AVAILABLE is True, the O(1) Rust compiled binary handles calculation.
        """
        t_rem = max(0.0, time_remaining)

        if RUST_AVAILABLE:
            try:
                # The Rust calculate_quotes returns (bid, ask) directly
                # Parameter order in Rust lib.rs get_quotes:
                # mid_price, volatility, inventory, risk_aversion, time_horizon, liquidity_param
                return rust_pricing.get_quotes(
                    mid_price,
                    self.sigma,
                    inventory_q,
                    self.gamma,
                    t_rem,
                    self.k
                )
            except Exception as e:
                logger.error(f"Rust execution failed: {e}. Falling back to Python.")

        # Fallback Python calculation
        r = self.get_reservation_price(mid_price, inventory_q, t_rem)
        delta = self.get_optimal_spread(t_rem)

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
