from __future__ import annotations
import numpy as np
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MarketScenario(BaseModel):
    """
    Represents a generated market scenario for stress testing.
    """
    scenario_id: str
    description: str
    risk_factors: Dict[str, float] = Field(..., description="Key risk indicators (e.g., 'inflation', 'unemployment')")
    probability_weight: float = Field(1.0, description="The likelihood of this scenario occurring")
    is_tail_event: bool = Field(False, description="Whether this represents a tail risk event")
    # v23.5 Additions (Optional to maintain backward compatibility)
    volatility_regime: Optional[str] = Field(None, description="Volatility state (Low/High/Jump)")
    correlation_id: Optional[str] = Field(None, description="ID of the correlation matrix used")

class GenerativeRiskEngine:
    """
    Implements a simplified Generative AI engine for Credit Risk Stress Testing.

    This class mimics the behavior of the GANs/VAEs described in 'The Quantum-AI Convergence
    in Credit Risk' (2025). In a full implementation, this would wrap a PyTorch model.
    Here, it provides the structural logic for 'Tail Enrichment' and 'Reverse Stress Testing'.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        # Mock latent space statistics for different regimes
        self.regimes = {
            "normal": {"mean": 0.0, "std": 1.0},
            "stress": {"mean": -2.0, "std": 2.5},
            "crash": {"mean": -4.0, "std": 4.0}
        }
        logger.info("Initialized GenerativeRiskEngine.")

    def generate_scenarios(self, n_samples: int = 1000, regime: str = "normal") -> List[MarketScenario]:
        """
        Generates synthetic market scenarios based on the requested regime.

        Args:
            n_samples (int): Number of scenarios to generate.
            regime (str): The market regime ('normal', 'stress', 'crash').

        Returns:
            List[MarketScenario]: A list of generated scenario objects.
        """
        if regime not in self.regimes:
            logger.warning(f"Unknown regime '{regime}', defaulting to 'normal'.")
            regime = "normal"

        params = self.regimes[regime]
        scenarios = []

        # Simulate generation process
        for i in range(n_samples):
            # In a real GAN, this would be: generator(noise_vector)
            inflation = np.random.normal(2.0, 1.0) + (abs(params["mean"]) if regime != "normal" else 0)
            unemployment = np.random.normal(4.0, 1.0) + (abs(params["mean"]) / 2 if regime != "normal" else 0)
            gdp_growth = np.random.normal(2.5, 1.5) + params["mean"]

            is_tail = gdp_growth < -2.0 or inflation > 8.0

            scenarios.append(MarketScenario(
                scenario_id=f"{regime}_{i}",
                description=f"Synthetic {regime} scenario",
                risk_factors={
                    "inflation": float(inflation),
                    "unemployment": float(unemployment),
                    "gdp_growth": float(gdp_growth)
                },
                probability_weight=1.0 / n_samples,
                is_tail_event=is_tail,
                volatility_regime=regime  # Enriched
            ))

        return scenarios

    def reverse_stress_test(self, target_loss_threshold: float, current_portfolio_value: float) -> List[MarketScenario]:
        """
        Performs Reverse Stress Testing using Conditional Generation.

        Identifies the exact market conditions that would cause a loss greater than the threshold.

        Args:
            target_loss_threshold (float): The loss amount to solve for.
            current_portfolio_value (float): Total portfolio value.

        Returns:
            List[MarketScenario]: Scenarios that trigger the breach.
        """
        logger.info(f"Running Reverse Stress Test for loss > {target_loss_threshold}")

        # We simulate a "Search" in the latent space for adverse conditions
        # In a real Conditional GAN, we would feed the 'loss' label to the generator.

        critical_scenarios = []

        # Generate a batch of 'crash' scenarios
        candidates = self.generate_scenarios(n_samples=500, regime="crash")

        for scen in candidates:
            # Simplified loss function (proxy model)
            # Loss increases if GDP drops and Unemployment rises
            impact = (4.0 - scen.risk_factors["gdp_growth"]) * 0.1 + \
                     (scen.risk_factors["unemployment"] - 4.0) * 0.05

            estimated_loss = current_portfolio_value * max(0, impact)

            if estimated_loss > target_loss_threshold:
                scen.description = f"BREACH DETECTED: Loss {estimated_loss:.2f}"
                critical_scenarios.append(scen)

        logger.info(f"Found {len(critical_scenarios)} scenarios that breach the threshold.")
        return critical_scenarios

# --- v23.5 Additive Evolution ---

class JumpDiffusionParams(BaseModel):
    """
    Parameters for the Merton Jump-Diffusion Model.
    Reference: Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous.
    """
    mu: float = Field(..., description="Drift rate (annualized)")
    sigma: float = Field(..., description="Volatility (annualized standard deviation)")
    lambda_j: float = Field(..., description="Jump intensity (expected jumps per year)")
    mu_j: float = Field(..., description="Mean of jump size")
    sigma_j: float = Field(..., description="Standard deviation of jump size")

class StochasticRiskEngine(GenerativeRiskEngine):
    """
    Apex-Level Risk Engine implementing stochastic calculus and correlated asset modeling.

    This engine upgrades the 'naive' generation of the base class with:
    1. Cholesky Decomposition for correlated risk factors.
    2. Merton Jump-Diffusion for heavy-tailed asset price simulation.
    3. Asynchronous execution for high-throughput scenario generation.
    """

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        logger.info("Initialized StochasticRiskEngine with advanced quantitative capabilities.")

    def _generate_correlated_noise(self, n_samples: int, n_vars: int, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Generates correlated Gaussian noise using Cholesky Decomposition.

        Math:
            L = Cholesky(Sigma)
            Z = Uncorrelated Normal Samples (n_samples x n_vars)
            X = Z @ L.T  -> Correlated Samples

        Args:
            n_samples (int): Number of samples.
            n_vars (int): Number of variables.
            correlation_matrix (np.ndarray): The correlation matrix (Sigma).

        Returns:
            np.ndarray: Correlated noise matrix.
        """
        # Validate matrix PSD
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            logger.error("Correlation matrix is not Positive Semi-Definite. Falling back to Identity.")
            L = np.eye(n_vars)

        uncorrelated_noise = np.random.normal(0, 1, size=(n_samples, n_vars))
        correlated_noise = uncorrelated_noise @ L.T
        return correlated_noise

    async def generate_scenarios_async(
        self,
        n_samples: int = 1000,
        regime: str = "normal",
        correlation_matrix: Optional[np.ndarray] = None
    ) -> List[MarketScenario]:
        """
        Asynchronously generates complex market scenarios with correlation structure.

        Args:
            n_samples (int): Number of samples.
            regime (str): Market regime.
            correlation_matrix (np.ndarray, optional): 3x3 correlation matrix for [Inflation, Unemployment, GDP].

        Returns:
            List[MarketScenario]: Enriched scenarios.
        """
        logger.info(f"Async generation started for {n_samples} samples in {regime} regime.")

        # Simulate compute intensity
        await asyncio.sleep(0.01)

        if correlation_matrix is None:
            # Default correlation: Inflation & Unemployment + (Phillips Curve), GDP & Unemployment - (Okun's Law)
            correlation_matrix = np.array([
                [1.0, 0.5, -0.3],  # Inflation
                [0.5, 1.0, -0.6],  # Unemployment
                [-0.3, -0.6, 1.0]  # GDP
            ])

        params = self.regimes.get(regime, self.regimes["normal"])

        # Generate correlated noise for 3 factors
        noise = self._generate_correlated_noise(n_samples, 3, correlation_matrix)

        scenarios = []
        for i in range(n_samples):
            # Scale and Shift based on regime
            # Factors: 0=Inflation, 1=Unemployment, 2=GDP
            inflation = noise[i, 0] * 1.0 + 2.0 + (abs(params["mean"]) if regime != "normal" else 0)
            unemployment = noise[i, 1] * 1.0 + 4.0 + (abs(params["mean"]) / 2 if regime != "normal" else 0)
            gdp_growth = noise[i, 2] * 1.5 + 2.5 + params["mean"]

            is_tail = gdp_growth < -2.5 or inflation > 9.0

            scenarios.append(MarketScenario(
                scenario_id=f"ASYNC_{regime}_{i}",
                description=f"Correlated {regime} scenario (Stochastic)",
                risk_factors={
                    "inflation": float(inflation),
                    "unemployment": float(unemployment),
                    "gdp_growth": float(gdp_growth)
                },
                probability_weight=1.0 / n_samples,
                is_tail_event=is_tail,
                volatility_regime=regime,
                correlation_id="standard_phillips_okun"
            ))

        return scenarios

    def simulate_merton_jump_diffusion(
        self,
        S0: float,
        T: float,
        dt: float,
        params: JumpDiffusionParams,
        n_paths: int = 100
    ) -> np.ndarray:
        """
        Simulates asset price paths using the Merton Jump-Diffusion Model.

        S_t = S_0 * exp((mu - 0.5*sigma^2 - lambda*k)t + sigma*W_t + sum(J_i))

        Where:
        - W_t is a Wiener process (Geometric Brownian Motion)
        - J_i is the jump size (Poisson process)

        Args:
            S0 (float): Initial stock price.
            T (float): Time horizon (years).
            dt (float): Time step size.
            params (JumpDiffusionParams): Model parameters.
            n_paths (int): Number of paths to simulate.

        Returns:
            np.ndarray: Simulated paths (n_paths x n_steps).
        """
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Pre-compute constants
        mu, sigma = params.mu, params.sigma
        lam, mu_j, sigma_j = params.lambda_j, params.mu_j, params.sigma_j

        # Expected jump size relative risk adjustment (k)
        # k = E[exp(J) - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Drift correction
        drift = (mu - 0.5 * sigma**2 - lam * k) * dt
        vol_step = sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # 1. Brownian Motion Component
            z = np.random.normal(0, 1, n_paths)

            # 2. Poisson Jump Component
            # Number of jumps in this step (Poisson distributed)
            n_jumps = np.random.poisson(lam * dt, n_paths)

            # Sum of Jumps for each path
            jump_factor = np.zeros(n_paths)
            # Optimization: Only process paths that have jumps
            has_jumps = n_jumps > 0
            if np.any(has_jumps):
                # We approximate multiple jumps in one step as sum of normals,
                # but for small dt, usually n_jumps is 0 or 1.
                # J ~ N(mu_j, sigma_j)
                # Total Jump = sum(J) ~ N(n * mu_j, sqrt(n) * sigma_j)
                j_mean = n_jumps[has_jumps] * mu_j
                j_std = np.sqrt(n_jumps[has_jumps]) * sigma_j
                jump_magnitude = np.random.normal(j_mean, j_std)
                jump_factor[has_jumps] = jump_magnitude

            # Update paths
            # S_t = S_{t-1} * exp(drift + diffusion + jumps)
            paths[:, t] = paths[:, t-1] * np.exp(drift + vol_step * z + jump_factor)

        return paths

# Example usage
if __name__ == "__main__":
    # Test Legacy
    engine = GenerativeRiskEngine()
    print(f"Legacy Sample: {engine.generate_scenarios(n_samples=1)[0]}")

    # Run reverse stress test (Legacy)
    breach_scenarios = engine.reverse_stress_test(target_loss_threshold=1000000, current_portfolio_value=10000000)
    if breach_scenarios:
        print(f"Legacy Critical Breach Scenario: {breach_scenarios[0]}")

    # Test Advanced
    advanced_engine = StochasticRiskEngine()

    # 1. Correlated Generation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    correlated = loop.run_until_complete(advanced_engine.generate_scenarios_async(n_samples=5, regime="stress"))
    print(f"\nAdvanced Correlated Sample: {correlated[0]}")

    # 2. Merton Jump Diffusion
    jd_params = JumpDiffusionParams(
        mu=0.05,
        sigma=0.2,
        lambda_j=0.5, # 1 jump every 2 years expected
        mu_j=-0.2,    # Jumps are downward shocks on average (-20%)
        sigma_j=0.1   # Volatility of jump size
    )

    paths = advanced_engine.simulate_merton_jump_diffusion(
        S0=100.0, T=1.0, dt=1/252, params=jd_params, n_paths=3
    )
    print(f"\nMerton Paths Shape: {paths.shape}")
    print(f"Final Prices: {paths[:, -1]}")
