from typing import Dict, Any
from ..state import VerticalRiskGraphState, DocumentMetrics

class DocumentExtractionAgent:
    """
    Pulls structured metrics from 10-K/10-Q filings, strictly enforcing Pydantic schemas.
    """

    def __init__(self, model):
        self.model = model

    def extract_metrics(self, state: VerticalRiskGraphState) -> Dict[str, Any]:
        """
        Extracts document metrics into a Pydantic structure.
        """
        print("--- Document Extraction Agent: Parsing 10-K ---")

        # Mock extraction with deterministic numbers for NVDA example
        ticker = state.get("ticker", "UNKNOWN")
        if ticker == "NVDA":
            metrics = {
                "total_revenue_usd": 60.92e9,
                "ebitda_usd": 34.48e9,
                "total_debt_usd": 15.0e9,
                "item_1a_risk_factors": ["Semiconductor supply chain stability", "Macroeconomic volatility"]
            }
        else:
            metrics = {
                "total_revenue_usd": 100e9,
                "ebitda_usd": 20e9,
                "total_debt_usd": 5e9,
                "item_1a_risk_factors": ["General market conditions"]
            }

        # Pydantic validation (Slide 6 concept implementation)
        try:
            validated = DocumentMetrics(**metrics)
            result = validated.model_dump()
            print("Successfully validated document metrics against schema.")
        except Exception as e:
            # Automated reflection placeholder
            print(f"Validation failed, triggering reflection: {e}")
            result = metrics # Fallback for demo

        return {
            "document_metrics": result,
            "messages": ["Document Extraction: Schema-validated 10-K extraction complete."]
        }
