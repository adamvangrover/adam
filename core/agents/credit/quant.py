import logging
from typing import Dict, Any, List

from core.agents.credit.credit_agent_base import CreditAgentBase

class QuantAgent(CreditAgentBase):
    """
    The Spreading Agent.
    Responsible for extracting structured financial data from unstructured documents.
    """

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: {'documents': [...], 'market_data': ...}
        Output: {'total_assets': ..., 'total_liabilities': ..., 'total_equity': ..., 'ratios': ...}
        """
        documents = inputs.get("documents", [])

        # In a real system, we'd OCR the tables here.
        # For this mock, we look for a special "financial_table" chunk type in the documents.
        financial_data = {}

        for doc in documents:
            for chunk in doc.get("chunks", []):
                if chunk.get("type") == "financial_table":
                    # Assume the chunk content is JSON or easily parsable
                    try:
                        table_data = chunk.get("content_json", {})
                        financial_data.update(table_data)
                    except Exception as e:
                        logging.warning(f"Failed to parse financial table chunk: {e}")

        # If no table found, return empty or mock defaults
        if not financial_data:
             logging.warning("No financial table found in documents.")
             return {}

        # Perform Checksum Validation (Assets = Liab + Equity)
        assets = financial_data.get("total_assets", 0.0)
        liabilities = financial_data.get("total_liabilities", 0.0)
        equity = financial_data.get("total_equity", 0.0)

        if abs(assets - (liabilities + equity)) > 1.0:
            logging.error(f"Balance Sheet Mismatch! Assets: {assets}, L+E: {liabilities + equity}")
            # We could raise an error or flag it
            financial_data["validation_error"] = "Balance Sheet Mismatch"
        else:
            financial_data["validation_status"] = "PASS"

        # Calculate Ratios
        ebitda = financial_data.get("ebitda", 0.0)
        debt = financial_data.get("total_debt", 0.0)
        interest = financial_data.get("interest_expense", 1.0) # Avoid div/0

        financial_data["ratios"] = {
            "leverage": debt / ebitda if ebitda else 0.0,
            "dscr": ebitda / interest if interest else 0.0
        }

        return financial_data
