import logging
from typing import Dict, Any, List
from core.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class PortfolioAggregator:
    """
    Asset Management capability.
    Aggregates risk and performance across multiple family entities/trusts.
    Integrates with Core Risk Engine for VaR analysis.
    """

    def __init__(self):
        self.risk_engine = RiskEngine()

    def aggregate_risk(self, portfolios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Unifies risk views and runs stress tests.
        """
        total_aum = sum(p.get("aum", 0) for p in portfolios)
        if total_aum == 0:
            return {"total_aum": 0, "risk_level": "N/A"}

        # Construct a synthetic "Master Portfolio" for the Risk Engine
        # Flatten structure
        master_positions = []
        for p in portfolios:
            # Assume portfolios have 'positions' or treat the portfolio itself as a single asset block
            # For this mock, we treat each portfolio as a single asset with an assumed vol
            master_positions.append({
                "symbol": p.get("name", "Unknown Fund"),
                "market_value": p.get("aum", 0),
                "volatility": p.get("implied_vol", 0.15)  # Default 15% vol
            })

        # Calculate VaR
        risk_metrics = self.risk_engine.calculate_portfolio_risk(master_positions)

        # Stress Tests
        stress_tests = {
            "Equity Crash (-20%)": total_aum * -0.20 * 0.6,  # Assuming 0.6 Beta
            "Rates Up (+100bps)": total_aum * -0.05  # Bond exposure proxy
        }

        return {
            "total_aum": total_aum,
            "entities_count": len(portfolios),
            "aggregated_risk_score": risk_metrics.get("ImpliedVol", 0) * 100,  # Scaled 0-100
            "daily_var_95": risk_metrics.get("VaR_Daily", 0),
            "top_exposures": [p.get("name") for p in portfolios],
            "stress_tests": stress_tests
        }
