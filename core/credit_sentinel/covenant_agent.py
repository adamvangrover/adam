import logging
from typing import Any, Dict, List, Optional, Tuple

from core.agents.templates.v30_template_agent import TemplateAgentV30
from core.schemas.agent_schema import AgentInput

logger = logging.getLogger("adam.credit_sentinel.covenant")


class CovenantAnalystAgent(TemplateAgentV30):
    """
    System 2 Agent responsible for evaluating debt covenants and calculating headroom.
    Strictly uses deterministic tools for calculations to prevent hallucination.
    """

    def __init__(self, mcp_client: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {"name": "credit_sentinel_covenant"})
        self.mcp = mcp_client

    async def analyze(self, input_data: AgentInput) -> Tuple[str, List[str], float, Dict[str, Any]]:
        try:
            # 1. Extract Financials from Context
            financials = input_data.context.get("financial_statements", {})
            ebitda = financials.get("EBITDA")
            interest_expense = financials.get("interest_expense")

            if not ebitda or not interest_expense:
                return ("Insufficient financial data to calculate covenant headroom.", [], 0.1, {"error": "Missing EBITDA or Interest Expense in context"})

            # 2. Call Deterministic MCP Tool (No LLM Math)
            fccr_result = await self.mcp.call_tool(
                tool_name="calculate_fccr_headroom",
                arguments={
                    "ebitda": ebitda,
                    "interest_expense": interest_expense,
                    "covenant_minimum": 2.0 # In production, pulled from SEC extraction tool
                }
            )

            # 3. Synthesize and Log PROV-O Lineage
            headroom_bps = fccr_result.get("headroom_bps")
            is_breached = fccr_result.get("is_breached")

            if is_breached:
                answer = f"CRITICAL: Covenant breach detected. FCCR is {fccr_result['fccr']}, below the 2.0x minimum."
                confidence = 0.95
            else:
                answer = f"Covenant compliant. FCCR stands at {fccr_result['fccr']} with {headroom_bps} bps of headroom."
                confidence = 0.90

            return (answer, ["10-K Line Items: EBITDA, Interest Expense"], confidence, {
                    "next_step": "consult_legal" if is_breached else "update_hdkg",
                    "hdkg_covenant_node": fccr_result
                })

        except Exception as e:
            logger.error(f"[Agent:Risk] Execution failed: {str(e)}")
            return ("System error during covenant evaluation. Failsafe triggered.", [], 0.0, {"trace": str(e)})
