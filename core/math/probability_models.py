"""
core/math/probability_models.py

Project OMEGA: Financial Math Library.
Provides statistical distributions for modeling market behavior, specifically focusing on
"Fat Tail" events (Black Swans) which are often underestimated by standard Gaussian models.
"""

import math
import random
import logging
from typing import List, Tuple, Dict, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

class ProbabilityModel:
    """Base class for statistical models."""
    def sample(self) -> float:
        raise NotImplementedError

    def pdf(self, x: float) -> float:
        """Probability Density Function"""
        raise NotImplementedError

class GaussianModel(ProbabilityModel):
    """
    Standard Normal Distribution.
    Used for: "Normal" market days, noise.
    """
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        if HAS_NUMPY:
            return np.random.normal(self.mu, self.sigma)
        return random.gauss(self.mu, self.sigma)

    def pdf(self, x: float) -> float:
        """
        1 / (sigma * sqrt(2*pi)) * exp( - (x - mu)^2 / (2 * sigma^2) )
        """
        denom = self.sigma * math.sqrt(2 * math.pi)
        num = math.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        return num / denom

class FatTailModel(ProbabilityModel):
    """
    Pareto / Cauchy Distribution.
    Used for: Crisis modeling, "Black Swan" events, crypto volatility.
    Generates extreme outliers more frequently than Gaussian.
    """
    def __init__(self, alpha: float = 3.0, xm: float = 1.0):
        """
        alpha: Shape parameter (Tail Index). Lower = fatter tails.
               Crypto ~ 2.0 - 3.0.
               S&P 500 ~ 3.0 - 4.0.
        xm: Scale parameter (Minimum value).
        """
        self.alpha = alpha
        self.xm = xm

    def sample(self) -> float:
        """
        Inverse Transform Sampling for Pareto.
        x = xm / (U)^(1/alpha)
        """
        u = random.random()
        return self.xm / math.pow(u, 1.0 / self.alpha)

    def sample_signed(self) -> float:
        """Returns positive or negative shock."""
        val = self.sample()
        sign = 1 if random.random() > 0.5 else -1
        return val * sign

class MarketRegimeModel:
    """
    Switching Model (HMM-like).
    Toggles between "Calm" (Gaussian) and "Chaos" (Fat Tail) based on a hidden state.
    """
    def __init__(self, transition_prob: float = 0.05):
        self.state = "CALM" # or "CHAOS"
        self.transition_prob = transition_prob
        self.gaussian = GaussianModel(mu=0.0005, sigma=0.01) # Daily return ~0.05%, Vol ~1%
        self.fat_tail = FatTailModel(alpha=2.5, xm=0.02)     # Min shock 2%

    def tick(self) -> Dict[str, Any]:
        """
        Simulate one time step.
        """
        # Transition logic
        if self.state == "CALM":
            if random.random() < self.transition_prob:
                self.state = "CHAOS"
        else:
            # Higher chance to revert to mean
            if random.random() < (self.transition_prob * 4):
                self.state = "CALM"

        # Sampling
        if self.state == "CALM":
            return_val = self.gaussian.sample()
            regime_vol = "LOW"
        else:
            # Chaos is usually negative in finance (crash), but can be melt-up
            shock = self.fat_tail.sample()
            # 80% chance of downside in chaos
            direction = -1 if random.random() < 0.8 else 1
            return_val = shock * direction
            regime_vol = "EXTREME"

        return {
            "regime": self.state,
            "return": return_val,
            "volatility_class": regime_vol
        }

if __name__ == "__main__":
    # Test
    print("--- Market Regime Simulation (10 Steps) ---")
    model = MarketRegimeModel(transition_prob=0.2) # High volatility for test
    for i in range(10):
        data = model.tick()
        fmt_ret = f"{data['return']*100:+.2f}%"
        print(f"T+{i}: [{data['regime']}] {fmt_ret} ({data['volatility_class']})")
