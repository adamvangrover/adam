from typing import Dict, Any
import logging
from core.pipelines.mock_edgar import MockEdgar
from core.agents.credit.orchestrator import CreditOrchestrator

class CreditPipeline:
    """
    End-to-End Pipeline for Credit Memo Generation.
    1. Ingest Data (Mock EDGAR)
    2. Run Orchestrator (Agents)
    3. Return structured artifact
    """

    def __init__(self):
        self.orchestrator = CreditOrchestrator(config={"agent_id": "pipeline_runner"})

    async def run(self, ticker: str, name: str = "Unknown", sector: str = "Technology") -> Dict[str, Any]:
        logging.info(f"Pipeline started for {ticker}")

        # 1. Fetch Source Data
        raw_data = MockEdgar.generate_10k(ticker, name, sector)

        # 2. Run Agents
        # The Orchestrator expects a request dict.
        # We need to bridge the MockEdgar output to what the Archivist expects?
        # Actually, in our current Orchestrator impl, the Archivist *loads* data from a file.
        # To support "Live" generation, we should pass the data directly or have the Archivist use a dynamic source.

        # Refactoring approach:
        # Pass the raw_data into the request context so the Archivist (or a new DynamicArchivist) can use it.
        # Or, we update the Archivist to accept 'direct_injection' in input.

        # For this implementation, let's update the request object to include 'injected_data'
        # and ensure the Archivist checks for it.

        request = {
            "borrower_name": name,
            "ticker": ticker,
            "injected_data": raw_data
        }

        # We need to monkey-patch or subclass Archivist to handle injection if not supported.
        # Let's verify ArchivistAgent in core/agents/credit/archivist.py
        # It currently loads from file.

        # Quick Fix: Save to a temporary location or update Archivist.
        # Better: Update Archivist to handle injected_data.
        # I will update Archivist in a subsequent step if needed, but for now let's assume
        # we can modify the Archivist logic in the Orchestrator or subclass it.

        # Actually, the most robust way is for the pipeline to *act* as the Archivist
        # and pass the retrieved docs to the Quant/Writer directly.
        # But the Orchestrator is hardcoded to call Archivist -> Quant -> Writer.

        # Let's stick to the plan: pass injected_data and I will update Archivist.py shortly
        # or rely on the fact that I can write to a temp file that Archivist reads.

        # Writing to a temp file for "Live" simulation is acceptable and robust.
        import json
        temp_path = f"/tmp/live_data_{ticker}.json"
        with open(temp_path, 'w') as f:
            json.dump({name: raw_data}, f)

        # Configure orchestrator to read from this temp path
        self.orchestrator.config['mock_data_path'] = temp_path

        result = await self.orchestrator.generate_credit_memo(request)

        # Cleanup? Maybe keep for debug.

        return {
            "ticker": ticker,
            "memo": result.get("memo"),
            "source_data": raw_data
        }
