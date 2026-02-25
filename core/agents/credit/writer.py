import logging
import asyncio
from typing import Dict, Any, List

from core.agents.credit.credit_agent_base import CreditAgentBase
from core.utils.prompt_loader import PromptLoader

class WriterAgent(CreditAgentBase):
    """
    The Generation Agent.
    Responsible for synthesizing the final credit memo and inserting citations.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prompt_loader = PromptLoader()
        self.config = config

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: {'borrower_name': ..., 'financial_context': ..., 'market_data': ...}
        Output: {'output': <memo_text>, 'citations': [...], 'confidence': ...}
        """
        try:
            prompt_config = self.prompt_loader.load_prompt("commercial_credit_risk_analysis")
        except Exception as e:
            logging.error(f"Failed to load prompt: {e}")
            return {"output": "Error loading prompt", "citations": [], "confidence": 0.0}

        # Prepare inputs for template
        # The prompt expects 'financial_context', 'market_data'
        financial_data = inputs.get("financial_context", {})

        # Format financial data into a string for the prompt
        fin_str = (
            f"Assets: {financial_data.get('total_assets', 'N/A')}\n"
            f"Liabilities: {financial_data.get('total_liabilities', 'N/A')}\n"
            f"Equity: {financial_data.get('total_equity', 'N/A')}\n"
            f"Validation: {financial_data.get('validation_status', 'Unknown')}"
        )

        # New: Pass LBO and Distressed Data
        lbo_data = inputs.get("lbo_analysis", "N/A")
        distressed_data = inputs.get("distressed_scenarios", "N/A")

        context = {
            "borrower_name": inputs.get("borrower_name", "Unknown Borrower"),
            "financial_context": fin_str,
            "market_data": inputs.get("market_data", "No market data."),
            "lbo_analysis": str(lbo_data),
            "distressed_scenarios": str(distressed_data)
        }

        messages = self.prompt_loader.render_messages(prompt_config, context)

        logging.info("Writer generating memo...")
        # Mock LLM Call
        memo = self._mock_llm_response(context)
        citations = self._extract_citations(memo)

        # Log Execution
        self.log_execution(
            inputs=context,
            output=memo,
            citations=citations,
            metadata={
                "prompt_version_id": prompt_config.version,
                "retrieved_chunks": [], # In real flow, pass IDs
                "graph_context": []
            }
        )

        return {
            "output": memo,
            "citations": citations,
            "confidence": 0.95,
            "metadata": {"prompt_version": prompt_config.version}
        }

    def _mock_llm_response(self, context: Dict[str, Any]) -> str:
        borrower = context.get("borrower_name", "The Borrower")
        fin = context.get("financial_context", "")
        lbo = context.get("lbo_analysis", "")
        distressed = context.get("distressed_scenarios", "")

        # Attempt to find a chunk ID to cite
        citation_1 = "[doc_123:chunk_456]"
        citation_2 = "[doc_123:chunk_789]"

        return f"""# Credit Memo: {borrower}

## Executive Summary
{borrower} presents a moderate risk profile. Revenue has grown by 15% YoY {citation_1}, primarily driven by expansion in the APAC region.

## Financial Analysis
{fin}
Leverage remains within covenant limits at 2.5x {citation_2}. However, liquidity ratios have tightened.

## Leveraged Finance / LBO
{lbo}

## Distressed Scenarios
{distressed}

## Recommendation
APPROVE with conditions."""

    def _extract_citations(self, text: str) -> List[str]:
        import re
        return re.findall(r"\[doc_.*?:.*?\]", text)
