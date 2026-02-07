from typing import Dict, Any, List
import logging
from core.agents.agent_base import AgentBase
from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
# Import MarketScenario from state to ensure consistency across the repo
from core.vertical_risk_agent.state import MarketScenario
from core.agents.mixins.audit_mixin import AuditMixin
from core.infrastructure.semantic_cache import SemanticCache

class DeepSectorAnalyst(AgentBase, AuditMixin):
    """
    A Deep Vertical Agent specialized in generating detailed sector-specific
    stress scenarios using the Generative Risk Engine.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        AuditMixin.__init__(self) # Explicit mixin init
        self.risk_engine = GenerativeRiskEngine()
        self.cache = SemanticCache()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Analyzes a sector exposure by generating regime-specific scenarios.
        """
        sector = kwargs.get("sector", "General")
        portfolio_value = kwargs.get("portfolio_value", 1_000_000.0)

        logging.info(f"DeepSectorAnalyst analyzing {sector}...")

        # Check Cache
        cache_key = self.cache.compute_data_hash({"sector": sector, "value": portfolio_value})
        cached = self.cache.get("DeepSectorAnalysis", cache_key, "v1_gen_risk")
        if cached:
            logging.info("DeepSectorAnalyst: Cache hit.")
            return cached

        # 1. Generate Stress Scenarios (Generative)
        scenarios = self.risk_engine.generate_scenarios(n_samples=50, regime="stress")

        # 2. Reverse Stress Test (What breaks us?)
        target_loss = portfolio_value * 0.20 # 20% loss threshold
        breaches = self.risk_engine.reverse_stress_test(target_loss, portfolio_value)

        # 3. Synthesize Insights
        tail_events = [s for s in scenarios if s.is_tail_event]

        result = {
            "sector": sector,
            "generated_scenarios_count": len(scenarios),
            "tail_risk_events_detected": len(tail_events),
            "reverse_stress_test_breaches": len(breaches),
            "critical_scenario_example": breaches[0].model_dump() if breaches else None,
            "recommendation": "Reduce exposure" if len(breaches) > 0 else "Maintain position"
        }

        # Log Decision
        self.log_decision(
            activity_type="SectorStressTest",
            details={"sector": sector, "regime": "stress"},
            outcome=result
        )

        # Cache Result
        self.cache.set("DeepSectorAnalysis", cache_key, "v1_gen_risk", result)

        return result

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "DeepSectorAnalyst",
            "description": "Generates deep vertical stress scenarios using Generative AI.",
            "skills": [
                {
                    "name": "analyze_sector_risk",
                    "description": "Runs generative stress tests on a sector exposure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sector": {"type": "string"},
                            "portfolio_value": {"type": "number"}
                        },
                        "required": ["sector"]
                    }
                }
            ]
        }
