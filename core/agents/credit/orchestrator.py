import asyncio
import logging
from typing import Dict, Any, List

from core.agents.credit.credit_agent_base import CreditAgentBase
# Assuming these will be created in the next step
# from core.agents.credit.archivist import ArchivistAgent
# from core.agents.credit.quant import QuantAgent
# from core.agents.credit.writer import WriterAgent

class CreditOrchestrator:
    """
    Orchestrates the Credit Memo generation process.
    Coordinates: Archivist -> Quant -> RiskOfficer -> Writer.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # We'll instantiate agents here or inject them
        # For now, we'll import them dynamically to avoid circular imports during setup
        pass

    async def generate_credit_memo(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for generating a credit memo.
        """
        logging.info(f"Starting Credit Memo generation for {request.get('borrower_name')}")

        # dynamic imports
        from core.agents.credit.archivist import ArchivistAgent
        from core.agents.credit.quant import QuantAgent
        from core.agents.credit.writer import WriterAgent

        archivist = ArchivistAgent(self.config)
        quant = QuantAgent(self.config)
        writer = WriterAgent(self.config)

        # 1. Retrieval Phase
        logging.info("Phase 1: Retrieval (Archivist)")
        docs = await archivist.execute(request)

        # 2. Spreading Phase
        logging.info("Phase 2: Spreading (Quant)")
        spreads = await quant.execute(docs)

        # 3. Generation Phase
        logging.info("Phase 3: Generation (Writer)")
        # The writer needs both the raw docs context and the structured spreads
        context = {
            "borrower_name": request.get("borrower_name"),
            "financial_context": spreads,
            "market_data": docs.get("market_data", "No market data found."),
            "docs": docs # Pass full docs for citations
        }

        memo = await writer.execute(context)

        return {
            "memo": memo,
            "audit_trail": "Generated via CreditOrchestrator"
        }
