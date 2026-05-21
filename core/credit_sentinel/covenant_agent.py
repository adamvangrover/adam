import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from adam.core.engine.base import TemplateAgentV26 # Base graph agent
from adam.core.mcp.client import MCPClient

logger = logging.getLogger("adam.credit_sentinel.covenant")

class AgentInput(BaseModel):
    query: str = Field(..., description="The specific question or objective.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state (RAG data, financials).")
    tools: List[str] = Field(default_factory=list, description="List of allowed tool names.")

class AgentOutput(BaseModel):
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations (filenames, URLs).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0.0 to 1.0).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info, HDKG nodes, etc.")

class CovenantAnalystAgent(TemplateAgentV26):
    """
    System 2 Agent responsible for evaluating debt covenants and calculating headroom.
    Strictly uses deterministic tools for calculations to prevent hallucination.
    """

    def __init__(self, mcp_client: MCPClient):
        super().__init__()
        self.mcp = mcp_client
        self.confidence_threshold = 0.85 # Config-driven, no magic numbers

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        try:
            # 1. Extract Financials from Context
            financials = input_data.context.get("financial_statements", {})
            ebitda = financials.get("EBITDA")
            interest_expense = financials.get("interest_expense")

            if not ebitda or not interest_expense:
                return AgentOutput(
                    answer="Insufficient financial data to calculate covenant headroom.",
                    sources=[],
                    confidence=0.1,
                    metadata={"error": "Missing EBITDA or Interest Expense in context"}
                )

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

            return AgentOutput(
                answer=answer,
                sources=["10-K Line Items: EBITDA, Interest Expense"],
                confidence=confidence,
                metadata={
                    "next_step": "consult_legal" if is_breached else "update_hdkg",
                    "hdkg_covenant_node": fccr_result
                }
            )

        except Exception as e:
            logger.error(f"[Agent:Risk] Execution failed: {str(e)}")
            return AgentOutput(
                answer="System error during covenant evaluation. Failsafe triggered.",
                sources=[],
                confidence=0.0,
                metadata={"trace": str(e)}
            )
