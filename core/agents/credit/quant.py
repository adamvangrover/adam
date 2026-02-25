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
        fcf = financial_data.get("fcf", 0.0) # Free Cash Flow
        capex = financial_data.get("capex", 0.0)

        # Enhanced Ratios for Leveraged Finance
        # Handle edge cases: 0 or negative EBITDA implies distressed/high leverage if debt exists
        if ebitda > 0:
            leverage = debt / ebitda
        elif debt > 0:
            leverage = 99.0 # Distressed / Infinite
        else:
            leverage = 0.0 # No debt, no earnings -> 0 leverage

        interest_coverage = ebitda / interest if interest else 0.0
        fccr = (ebitda - capex) / interest if interest else 0.0 # Fixed Charge Coverage Ratio (simplified)

        financial_data["ratios"] = {
            "leverage": leverage,
            "interest_coverage": interest_coverage,
            "fccr": fccr,
            "dscr": ebitda / interest if interest else 0.0
        }

        # Structure Debt Tranches (Mock extraction)
        # In real OCR, we'd parse "Senior Notes", "Term Loan B", etc.
        # Here we look for them in the data or infer defaults.
        financial_data["debt_structure"] = {
            "senior_secured": financial_data.get("senior_secured_debt", debt * 0.6),
            "unsecured": financial_data.get("unsecured_debt", debt * 0.4),
            "subordinated": financial_data.get("subordinated_debt", 0.0)
        }

        return financial_data
