import logging
import asyncio
from typing import Dict, Any, List, Optional
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import QuantumScenario

# Try importing the real engines, fallback to None if dependencies missing
try:
    from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine
    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False

try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
    GRE_AVAILABLE = True
except ImportError:
    GRE_AVAILABLE = False

logger = logging.getLogger(__name__)

class QuantumScenarioAgent(AgentBase):
    """
    Phase 4 Helper: Quantum Scenario Generation.

    This agent bridges the gap between classical risk modeling and quantum-enhanced simulation.
    It utilizes the `QuantumMonteCarloEngine` (QMC) for structural credit modeling and the
    `GenerativeRiskEngine` (GRE) for tail-risk scenario generation.

    Developer Note:
    ---------------
    In environments without a QPU or heavy GPU dependencies, this agent gracefully degrades
    to use classical approximations (numpy-based QMC simulation) and heuristic-based
    generative models.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Quantum Risk Physicist"

        # Initialize Engines if available
        self.qmc_engine = QuantumMonteCarloEngine() if QMC_AVAILABLE else None
        self.gre_engine = GenerativeRiskEngine(mode="inference") if GRE_AVAILABLE else None

    async def execute(self, **kwargs) -> List[QuantumScenario]:
        """
        Executes the quantum scenario generation pipeline.

        Args:
            ticker (str): The entity identifier.
            market_data (Dict): Optional market context.
            financials (Dict): Optional financial data for Merton model parameters.
        """
        logger.info("Executing Quantum Scenario Generation...")

        # Support both nested 'params' key and flat kwargs
        params = kwargs.get('params', kwargs)
        ticker = params.get('ticker', 'Unknown')
        financials = params.get('financials', {})

        scenarios = []

        # 1. Quantum Monte Carlo (Structural Model)
        # -----------------------------------------
        if self.qmc_engine and financials:
            try:
                # Extract Merton Parameters or use defaults
                asset_value = float(financials.get('total_assets', 100.0))
                debt_value = float(financials.get('total_debt', 50.0))
                volatility = float(financials.get('volatility', 0.2))

                # Run Simulation
                qmc_result = self.qmc_engine.simulate_merton_model(
                    asset_value=asset_value,
                    debt_face_value=debt_value,
                    volatility=volatility,
                    risk_free_rate=0.04,
                    time_horizon=1.0
                )

                # Convert QMC result to a Scenario
                pd = qmc_result.get('probability_of_default', 0.0)
                scenarios.append(QuantumScenario(
                    name="QMC: Structural Default Path",
                    probability=pd,
                    estimated_impact_ev="100% Equity Wipeout",
                    description=f"Quantum Amplitude Estimation predicts {pd:.2%} default probability based on asset volatility of {volatility:.0%}."
                ))
            except Exception as e:
                logger.error(f"QMC Engine failed: {e}")

        # 2. Generative Risk Engine (Tail Events)
        # ---------------------------------------
        if self.gre_engine:
            try:
                # We need a mock 'MacroEvent' to trigger the generator
                from core.schemas.v23_5_schema import MacroEvent
                trigger = MacroEvent(name="Rate Shock", type="inflation", description="Unexpected 50bps hike", impact_score=0.8)

                gre_scenarios = self.gre_engine.generate_stress_scenario(trigger, n_scenarios=2)

                # Convert CrisisScenario to QuantumScenario (mapping fields)
                for gs in gre_scenarios:
                    scenarios.append(QuantumScenario(
                        name=f"GRE: {gs.name}",
                        probability=gs.probability,
                        estimated_impact_ev=f"-{gs.severity_score*100:.0f}%",
                        description=gs.description
                    ))
            except Exception as e:
                 logger.warning(f"GRE Engine generation failed or schema mismatch: {e}")

        # 3. Fallback / Augmentation Logic
        # --------------------------------
        # If engines didn't produce enough scenarios, use advanced heuristics
        if len(scenarios) < 2:
            logger.info("Augmenting with heuristic scenarios due to engine constraints.")
            scenarios.extend(self._generate_heuristic_scenarios(ticker))

        return scenarios

    def _generate_heuristic_scenarios(self, ticker: str) -> List[QuantumScenario]:
        """
        Generates plausible "Unknown Unknowns" when deep simulation is unavailable.
        """
        return [
            QuantumScenario(
                name="Geopolitical Flashpoint (Taiwan Straits)",
                probability=0.042, # Specific non-normal prob
                estimated_impact_ev="-40%",
                description=f"Supply chain blockade affecting {ticker} sourcing and logistics."
            ),
            QuantumScenario(
                name="Technological Singularity (AGI)",
                probability=0.08,
                estimated_impact_ev="+200%",
                description=f"Rapid automation of {ticker}'s core business processes leading to hyper-scaling."
            )
        ]
