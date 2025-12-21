# core/engine/agent_adapters.py

"""
Agent Notes (Meta-Commentary):
Provides a clean interface layer (Adapter Pattern) between the new v23 Graph Engine
and the legacy v21/v22 Agents. This ensures dependency isolation and backward compatibility.
"""

import json
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import Legacy Agents safely
try:
    from core.agents.risk_assessment_agent import RiskAssessmentAgent as LegacyRAA
except ImportError:
    LegacyRAA = None


class V23DataRetrieverAdapter:
    """
    Adapts DataRetrievalAgent logic for v23.
    Removes dependencies on Semantic Kernel/AsyncAgentBase where not needed.
    """

    def get_financials(self, company_id: str) -> Optional[Dict[str, Any]]:
        # Logic ported from core/agents/data_retrieval_agent.py
        # We simulate fetching data.
        # Robust mock for all tickers in this scaffolding phase
        return {
            "company_info": {
                "name": f"{company_id} Corp",
                "industry_sector": "Technology",
                "country": "USA"
            },
            "financial_data_detailed": {
                "income_statement": {
                    "revenue": [1000, 1100, 1250], "cogs": [400, 440, 500], "gross_profit": [600, 660, 750],
                    "operating_expenses": [300, 320, 350], "ebitda": [300, 340, 400], "depreciation_amortization": [50, 55, 60],
                    "ebit": [250, 285, 340], "interest_expense": [30, 28, 25], "income_before_tax": [220, 257, 315],
                    "taxes": [44, 51, 63], "net_income": [176, 206, 252]
                },
                "balance_sheet": {
                    "cash_and_equivalents": [200, 250, 300], "accounts_receivable": [150, 160, 170], "inventory": [100, 110, 120],
                    "total_current_assets": [450, 520, 590], "property_plant_equipment_net": [1500, 1550, 1600],
                    "total_assets": [1950, 2070, 2190],
                    "accounts_payable": [120, 130, 140], "short_term_debt": [100, 80, 60], "total_current_liabilities": [220, 210, 200],
                    "long_term_debt": [500, 450, 400], "total_liabilities": [720, 660, 600],
                    "shareholders_equity": [1230, 1410, 1590]
                },
                "key_ratios": {
                    "debt_to_equity_ratio": 0.58, "net_profit_margin": 0.20,
                    "current_ratio": 2.95, "interest_coverage_ratio": 13.6
                },
                "market_data": {
                    "share_price": 65.00, "shares_outstanding": 10000000
                }
            }
        }


class V23RiskAssessorAdapter:
    """
    Adapts RiskAssessmentAgent for v23.
    """

    def __init__(self):
        self.agent = LegacyRAA() if LegacyRAA else None

    def assess_investment_risk(self, ticker: str, financials: Dict, market: Dict) -> Dict:
        if not self.agent:
            return {"overall_risk_score": 0.0, "risk_factors": {"error": "Agent unavailable"}}

        try:
            return self.agent.assess_investment_risk(ticker, financials, market)
        except Exception as e:
            logger.error(f"RiskAssessmentAgent failed: {e}")
            # Check if it's the knowledge base error
            if "Knowledge base" in str(e):
                return {"overall_risk_score": 0.5, "risk_factors": {"error": "Knowledge Base Missing (Mocking Assessment)"}}
            return {"overall_risk_score": 0.0, "risk_factors": {"error": str(e)}}


def map_dra_to_raa(financials: Dict[str, Any]) -> tuple[Dict, Dict]:
    """
    Maps DataRetrievalAgent output to RiskAssessmentAgent input.
    """
    fin_details = financials.get("financial_data_detailed", {})

    mapped_fin = {
        "industry": financials.get("company_info", {}).get("industry_sector", "Unknown"),
        "credit_rating": "BBB",
    }

    current_price = fin_details.get("market_data", {}).get("share_price", 100)
    mapped_market = {
        "price_data": np.array([current_price * (1 + i*0.01) for i in range(10)])
    }

    return mapped_fin, mapped_market
