import json
import logging
from typing import Dict, Any, List

from core.agents.credit.credit_agent_base import CreditAgentBase

class ArchivistAgent(CreditAgentBase):
    """
    The Retrieval Agent.
    Responsible for fetching documents and chunks from the Vector DB (mocked).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Load from both mock files to support legacy tests and new demo
        self.mock_data_paths = [
            config.get("mock_data_path", "showcase/data/credit_mock_data.json"),
            "showcase/data/tech_credit_data.json"
        ]

    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        borrower_name = request.get("borrower_name")
        logging.info(f"Archivist retrieving data for: {borrower_name}")

        combined_data = {}

        # Merge data from all sources
        for path in self.mock_data_paths:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    combined_data.update(data)
            except FileNotFoundError:
                logging.warning(f"Mock data file not found: {path}")

        try:
            # Simple mock retrieval: Return everything for the borrower
            # In a real system, we'd do vector search here
            borrower_data = combined_data.get(borrower_name, {})

            if not borrower_data:
                logging.warning(f"No data found for {borrower_name}")
                return {"documents": [], "market_data": "No data found."}

            return {
                "documents": borrower_data.get("documents", []),
                "market_data": borrower_data.get("market_data", ""),
                "knowledge_graph": borrower_data.get("knowledge_graph", {})
            }

        except Exception as e:
            logging.error(f"Error retrieving data: {e}")
            return {"documents": [], "market_data": "Error loading data."}
