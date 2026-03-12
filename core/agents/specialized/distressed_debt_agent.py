import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class DebtTranche(BaseModel):
    facility_name: str
    amount_outstanding: float = Field(..., description="Amount outstanding in millions")
    lien_position: int = Field(..., description="1 for First Lien, 2 for Second Lien, etc.")
    recovery_estimate_pct: float = Field(..., description="Estimated recovery rate percentage (0.0 to 1.0)")
    covenant_breach_probability: float = Field(..., description="Probability of breaching covenants (0.0 to 1.0)")

class DistressedDebtAnalysis(BaseModel):
    issuer_name: str
    enterprise_value_distressed: float = Field(..., description="Estimated EV in a distressed scenario in millions")
    implied_default_probability: float = Field(..., description="Market-implied probability of default (0.0 to 1.0)")
    tranches: List[DebtTranche]
    restructuring_strategy: str = Field(..., description="Recommended restructuring approach (e.g., Amend & Extend, Equitization, Chapter 11)")

class DistressedDebtAgent(AgentBase):
    """
    DistressedDebtAgent is responsible for analyzing deep credit risk and distressed debt scenarios.
    It evaluates recovery rates across different debt tranches, models default probabilities,
    and proposes restructuring strategies for Leveraged Finance underwriting.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        super().__init__(config, kernel=kernel)
        self.agent_name = "DistressedDebtAgent"

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the distressed debt analysis workflow.

        Expected kwargs:
            issuer_name (str): Name of the borrower.
            financials (Dict): Historical financial data.
            market_data (Dict): Current market pricing data.
            debt_structure (List[Dict]): Outstanding debt facilities.
        """
        issuer_name = kwargs.get("issuer_name", "Unknown Issuer")
        financials = kwargs.get("financials", {})
        debt_structure = kwargs.get("debt_structure", [])

        logger.info(f"[{self.agent_name}] Analyzing distressed scenarios for {issuer_name}")

        try:
            # Construct analysis prompt
            prompt = self._build_analysis_prompt(issuer_name, financials, debt_structure)

            # Request structured output from LLM plugin
            logger.info(f"[{self.agent_name}] Requesting structured distressed analysis from LLM")
            structured_data, _ = self.llm_plugin.generate_structured(
                prompt=prompt,
                schema_model=DistressedDebtAnalysis,
                task="distressed_analysis"
            )

            # If the LLM generates incomplete output or we are in a fallback mode, provide graceful defaults
            if not structured_data or not hasattr(structured_data, "tranches"):
                logger.warning(f"[{self.agent_name}] LLM output incomplete, falling back to heuristic model.")
                structured_data = self._heuristic_fallback(issuer_name, debt_structure)

            return {
                "status": "success",
                "issuer": issuer_name,
                "distressed_analysis": structured_data.model_dump()
            }

        except Exception as e:
            logger.error(f"[{self.agent_name}] Execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "issuer": issuer_name,
                "distressed_analysis": self._heuristic_fallback(issuer_name, debt_structure).model_dump()
            }

    def _build_analysis_prompt(self, issuer: str, financials: Dict, debt: List[Dict]) -> str:
        """Constructs the LLM prompt based on provided inputs."""
        ebitda = financials.get("ebitda", 0)
        debt_str = "\n".join([f"- {d.get('facility_name', 'Debt')}: ${d.get('amount', 0)}M (Lien: {d.get('lien', 1)})" for d in debt])

        return (
            f"You are a Distressed Debt Underwriting AI. Analyze the credit profile for '{issuer}'.\n"
            f"Financials context: EBITDA is approximately ${ebitda}M.\n"
            f"Capital Structure:\n{debt_str}\n\n"
            f"Calculate the Distressed Enterprise Value, implied default probability, and estimate "
            f"recovery rates and covenant breach probabilities for each tranche. Finally, suggest a restructuring strategy."
        )

    def _heuristic_fallback(self, issuer: str, debt: List[Dict]) -> DistressedDebtAnalysis:
        """Provides a deterministic fallback analysis if LLM fails."""
        tranches = []
        for d in debt:
            lien = d.get("lien", 1)
            recovery = 0.80 if lien == 1 else 0.40 if lien == 2 else 0.10
            tranches.append(
                DebtTranche(
                    facility_name=d.get("facility_name", f"Lien {lien} Debt"),
                    amount_outstanding=float(d.get("amount", 100)),
                    lien_position=lien,
                    recovery_estimate_pct=recovery,
                    covenant_breach_probability=0.25 * lien
                )
            )

        # Default scenario if no debt provided
        if not tranches:
            tranches.append(DebtTranche(
                facility_name="First Lien Term Loan",
                amount_outstanding=500.0,
                lien_position=1,
                recovery_estimate_pct=0.75,
                covenant_breach_probability=0.35
            ))

        return DistressedDebtAnalysis(
            issuer_name=issuer,
            enterprise_value_distressed=sum(t.amount_outstanding for t in tranches) * 0.8,
            implied_default_probability=0.15,
            tranches=tranches,
            restructuring_strategy="Amend & Extend"
        )
