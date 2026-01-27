import logging
import numpy as np
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import SimulationEngine, TradingDynamics, QuantumScenario

# Configure logging
logger = logging.getLogger(__name__)


class MonteCarloRequest(BaseModel):
    """
    Validates input parameters for the Monte Carlo Risk Agent.
    Supports GBM (default), Heston, and OU models.
    """
    model_config = ConfigDict(extra='ignore')

    # Financials
    current_ebitda: float = Field(..., description="Current TTM EBITDA")
    interest_expense: float = Field(0.0, ge=0.0, description="Annual Interest Expense")
    capex_maintenance: float = Field(0.0, ge=0.0, description="Maintenance Capex")

    # Simulation Configuration
    model_type: Literal["GBM", "Heston", "OU"] = Field("GBM", description="Stochastic model selection")
    iterations: int = Field(10000, ge=1000, description="Number of simulation paths")
    time_horizon: float = Field(1.0, gt=0.0, description="Time horizon in years")
    dt_steps: int = Field(12, ge=1, description="Number of time steps (12=monthly)")

    # Stochastic Parameters (GBM / Generic)
    drift: float = Field(0.02, description="Drift (mu) for GBM")
    volatility: float = Field(0.2, ge=0.0, description="Volatility (sigma)")

    # Heston Specific
    heston_kappa: float = Field(2.0, ge=0.0, description="Mean reversion speed of variance")
    heston_theta: float = Field(0.04, ge=0.0, description="Long-run variance")
    heston_xi: float = Field(0.3, ge=0.0, description="Volatility of volatility")
    heston_rho: float = Field(-0.5, ge=-1.0, le=1.0, description="Correlation (asset vs vol)")
    heston_v0: float = Field(0.04, ge=0.0, description="Initial variance")

    # Ornstein-Uhlenbeck Specific
    ou_theta: float = Field(0.15, ge=0.0, description="Mean reversion speed")
    ou_mu: float = Field(100.0, description="Long run mean level")


class MonteCarloRiskAgent(AgentBase):
    """
    Quantitative Risk Agent using Monte Carlo simulations.

    Methodology:
    1. Models EBITDA as a stochastic process (Geometric Brownian Motion, Heston, or OU).
    2. Runs iterations (default 10,000) over a defined horizon.
    3. Triggers 'Default' if EBITDA falls below Interest Expense + Maintenance Capex.

    Developer Note:
    ---------------
    Now supports Heston (stochastic volatility) and OU (mean reversion).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Quantitative Risk Modeler"
        # Defaults are now handled in Pydantic, but we can store global overrides here if needed
        self.iterations = 10000

    async def execute(self, **kwargs) -> SimulationEngine:
        """
        Runs the Monte Carlo simulation.

        Args (in kwargs):
            current_ebitda: TTM EBITDA.
            ebitda_volatility: Annualized standard deviation of EBITDA (e.g., 0.15 for 15%).
            interest_expense: Annual fixed interest cost.
            capex_maintenance: Essential capex required to keep business running.
            model_type: "GBM", "Heston", or "OU".

        Returns:
            SimulationEngine: Schema object with default probability and scenario placeholders.
        """
        logger.info(f"Running Monte Carlo Simulation...")

        # Map legacy keys to new schema if necessary, or rely on caller passing correct keys.
        # Support alias 'ebitda_volatility' -> 'volatility'
        if "ebitda_volatility" in kwargs and "volatility" not in kwargs:
            kwargs["volatility"] = kwargs["ebitda_volatility"]

        try:
            # Pydantic Validation
            request = MonteCarloRequest(**kwargs)
        except Exception as e:
            logger.error(f"Validation Error: {e}")
            # Fallback for critical failure (e.g. missing EBITDA)
            return SimulationEngine(
                monte_carlo_default_prob="N/A",
                quantum_scenarios=[],
                trading_dynamics=TradingDynamics(
                    short_interest="N/A", liquidity_risk="High"
                )
            )

        logger.info(f"Model: {request.model_type}, Iterations: {request.iterations}")

        # Input Validation / Fallbacks
        if request.current_ebitda == 0:
            logger.warning("Monte Carlo: EBITDA is 0. Returning Default.")
            return SimulationEngine(
                monte_carlo_default_prob="100.0%",
                quantum_scenarios=[],
                trading_dynamics=TradingDynamics(
                    short_interest="N/A", liquidity_risk="High"
                )
            )

        # 1. Define Distress Threshold (Cash Flow Solvency)
        distress_threshold = request.interest_expense + request.capex_maintenance

        # 2. Vectorized Simulation (NumPy)
        T = request.time_horizon
        dt = T / request.dt_steps

        # Route to appropriate model
        if request.model_type == "Heston":
            simulated_paths = self._run_heston_simulation(request, dt)
        elif request.model_type == "OU":
            simulated_paths = self._run_ou_simulation(request, dt)
        else:
            # Default to GBM
            simulated_paths = self._run_gbm_simulation(request, dt)

        # We take the terminal value for the simplified check, or check minimum over path?
        # The original code used terminal value check. For consistency with description:
        # "Triggers 'Default' if EBITDA falls below Interest Expense..."
        # Often this checks at any point (barrier) or at maturity.
        # Given the legacy code did "simulated_ebitda = ... (T=1)" (terminal), I will stick to terminal
        # for GBM single step, but since we now have dt_steps, let's check minimum over the path (American barrier)
        # or just terminal. Let's stick to terminal for now to match the "solvency at horizon" logic,
        # or better, check if it dips below at any measured point (more conservative/defensive).

        # Let's use terminal value to start, as implied by the previous "T=1" logic.
        # Actually, let's check the minimum value across the path for a more rigorous "bank-grade" risk check.
        # If min(EBITDA_t) < Threshold, then Default.

        min_ebitda_per_path = np.min(simulated_paths, axis=1)
        breaches = np.sum(min_ebitda_per_path < distress_threshold)

        pd_ratio = breaches / request.iterations
        logger.info(f"Simulation Result: {breaches} breaches in {request.iterations} paths. PD: {pd_ratio:.2%}")

        # 3. Construct Output
        if pd_ratio < 0.01:
            impact = "Minimal Impact"
        elif pd_ratio < 0.05:
            impact = "Equity Dilution Risk"
        else:
            impact = "Debt Restructuring Likely"

        def get_prob_bucket(p):
            if p < 0.1:
                return "Low"
            if p < 0.4:
                return "Med"
            return "High"

        # ---------------------------------------------------------
        # Bank-Grade Enhancements:
        # 1. Path Sampling (for visualization)
        # 2. Percentile Cones (1, 5, 50, 95, 99)
        # 3. Risk Metrics (VaR, CVaR) on Terminal Distribution
        # ---------------------------------------------------------

        # 1. Path Sampling (Take first 50 paths)
        sample_size = min(50, request.iterations)
        sampled_paths = simulated_paths[:sample_size, :].tolist()

        # 2. Percentiles over time
        # shape: (N, M+1) -> percentiles across axis 0
        percentile_levels = [1, 5, 50, 95, 99]
        percentiles_over_time = np.percentile(simulated_paths, percentile_levels, axis=0)
        # Convert to dict: "p5": [val_t0, val_t1, ...]
        percentile_dict = {
            f"p{p}": percentiles_over_time[i].tolist()
            for i, p in enumerate(percentile_levels)
        }

        # 3. Risk Metrics on Terminal Distribution (Horizon)
        terminal_values = simulated_paths[:, -1]

        # VaR (Value at Risk) - e.g. 5th percentile worst outcome
        # Since these are absolute values, VaR_95 is the 5th percentile value.
        var_95_val = np.percentile(terminal_values, 5)
        var_99_val = np.percentile(terminal_values, 1)

        # CVaR (Conditional VaR) - Mean of values below VaR
        cvar_95_val = terminal_values[terminal_values <= var_95_val].mean()
        cvar_99_val = terminal_values[terminal_values <= var_99_val].mean()

        simulation_metadata = {
            "model_type": request.model_type,
            "iterations": request.iterations,
            "time_horizon": request.time_horizon,
            "dt_steps": request.dt_steps,
            "sampled_paths": sampled_paths,
            "percentiles": percentile_dict,
            "risk_metrics": {
                "VaR_95": float(var_95_val),
                "VaR_99": float(var_99_val),
                "CVaR_95": float(cvar_95_val),
                "CVaR_99": float(cvar_99_val),
                "Distress_Threshold": float(distress_threshold)
            }
        }

        output = SimulationEngine(
            monte_carlo_default_prob=f"{pd_ratio:.1%}",
            quantum_scenarios=[
                QuantumScenario(
                    scenario_name=f"Base Case ({request.model_type})",
                    probability=get_prob_bucket(1.0 - pd_ratio),
                    impact_severity="Moderate",
                    estimated_impact_ev="Neutral"
                ),
                QuantumScenario(
                    scenario_name="Tail Risk (Default)",
                    probability=get_prob_bucket(pd_ratio),
                    impact_severity="Critical",
                    estimated_impact_ev=impact
                )
            ],
            trading_dynamics=TradingDynamics(
                short_interest="See Market Data",
                liquidity_risk="High" if request.volatility > 0.3 else "Low"
            ),
            simulation_metadata=simulation_metadata
        )

        return output

    def _run_gbm_simulation(self, request: MonteCarloRequest, dt: float) -> np.ndarray:
        """
        Geometric Brownian Motion.
        S_{t+1} = S_t * exp((mu - 0.5*sigma^2)dt + sigma * sqrt(dt) * Z)
        """
        N = request.iterations
        M = request.dt_steps
        S0 = request.current_ebitda
        mu = request.drift
        sigma = request.volatility

        # Paths matrix: (Iterations, Steps + 1)
        paths = np.zeros((N, M + 1))
        paths[:, 0] = S0

        for t in range(1, M + 1):
            Z = np.random.standard_normal(N)
            # Vectorized update
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        return paths

    def _run_heston_simulation(self, request: MonteCarloRequest, dt: float) -> np.ndarray:
        """
        Heston Stochastic Volatility Model.
        dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^S
        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v
        """
        N = request.iterations
        M = request.dt_steps
        S0 = request.current_ebitda

        mu = request.drift
        kappa = request.heston_kappa
        theta = request.heston_theta
        xi = request.heston_xi
        rho = request.heston_rho
        v0 = request.heston_v0

        # Arrays for Price (S) and Volatility (v)
        S = np.zeros((N, M + 1))
        v = np.zeros((N, M + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        for t in range(1, M + 1):
            # Generate Correlated Brownian Motions
            Z1 = np.random.standard_normal(N)
            Z2 = np.random.standard_normal(N)

            Z_S = Z1
            Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

            # Previous values
            S_prev = S[:, t-1]
            v_prev = v[:, t-1]

            # Ensure variance stays non-negative for the sqrt calculation (Reflection/Truncation)
            # Using Full Truncation: sqrt(max(0, v_prev))
            v_prev_plus = np.maximum(v_prev, 0.0)

            # Volatility Process (v_t)
            # dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
            dv = kappa * (theta - v_prev_plus) * dt + xi * np.sqrt(v_prev_plus * dt) * Z_v
            v[:, t] = v_prev + dv

            # Price Process (S_t)
            # dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_S
            dS = mu * S_prev * dt + np.sqrt(v_prev_plus * dt) * S_prev * Z_S
            S[:, t] = S_prev + dS

        return S

    def _run_ou_simulation(self, request: MonteCarloRequest, dt: float) -> np.ndarray:
        """
        Ornstein-Uhlenbeck Process (Mean Reversion).
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        """
        N = request.iterations
        M = request.dt_steps
        X0 = request.current_ebitda

        # Note: In OU, 'mu' is the long-run mean level, 'theta' is speed.
        # We mapped these in Pydantic schema: ou_mu -> Long Run Mean, ou_theta -> Speed
        long_run_mean = request.ou_mu
        theta = request.ou_theta
        sigma = request.volatility

        X = np.zeros((N, M + 1))
        X[:, 0] = X0

        for t in range(1, M + 1):
            Z = np.random.standard_normal(N)

            X_prev = X[:, t-1]

            # Update: X_{t+1} = X_t + theta * (mu - X_t) * dt + sigma * sqrt(dt) * Z
            dX = theta * (long_run_mean - X_prev) * dt + sigma * np.sqrt(dt) * Z

            X[:, t] = X_prev + dX

        return X
