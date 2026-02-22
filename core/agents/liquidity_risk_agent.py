from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import numpy as np
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class LiquidityRiskAgent(AgentBase):
    """
    Agent responsible for assessing Liquidity Risk.
    Calculates Financial Liquidity (Current/Quick Ratio, LCR)
    and Market Liquidity (Spread, Impact Cost).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the liquidity risk assessment.

        Args:
            data: Dictionary containing:
                - financial_data: {
                    current_assets, current_liabilities, inventory,
                    cash_and_equivalents, government_bonds, corporate_bonds_high_grade,
                    net_cash_outflows_30d (optional)
                  }
                - market_data: {
                    avg_daily_volume, bid_ask_spread, price, volatility
                  }
                - trade_size_simulation: float (optional, default 100000)

        Returns:
            Dict containing liquidity risk metrics.
        """
        logger.info("Starting Liquidity Risk Assessment...")

        financial_data = data.get("financial_data", {})
        market_data = data.get("market_data", {})
        trade_size = data.get("trade_size_simulation", 100000.0)

        result = {}

        # 1. Financial Liquidity (Balance Sheet)
        try:
            ca = financial_data.get("current_assets", 0)
            cl = financial_data.get("current_liabilities", 1)
            inv = financial_data.get("inventory", 0)

            if cl == 0: cl = 1

            current_ratio = ca / cl
            quick_ratio = (ca - inv) / cl

            # LCR Approximation
            lcr = self._calculate_lcr(financial_data)

            result["financial_liquidity"] = {
                "current_ratio": float(current_ratio),
                "quick_ratio": float(quick_ratio),
                "liquidity_coverage_ratio_approx": float(lcr),
                "assessment": "Good" if lcr > 1.0 else "Concern" if lcr < 0.8 else "Adequate"
            }
        except Exception as e:
            logger.error(f"Error calculating financial liquidity: {e}")
            result["financial_liquidity_error"] = str(e)

        # 2. Market Liquidity (Trading)
        try:
            avg_volume = market_data.get("avg_daily_volume", 0)
            bid_ask_spread = market_data.get("bid_ask_spread", 0)
            price = market_data.get("price", 100)

            # Spread percentage
            spread_pct = (bid_ask_spread / price) if price > 0 else 0

            # Market Impact Cost
            impact_cost = self._calculate_impact_cost(market_data, trade_size)

            result["market_liquidity"] = {
                "avg_daily_volume": float(avg_volume),
                "bid_ask_spread_pct": float(spread_pct),
                "simulated_impact_cost_pct": float(impact_cost),
                "simulated_trade_size": float(trade_size),
                "assessment": "Liquid" if impact_cost < 0.01 else "Illiquid" if impact_cost > 0.05 else "Moderate"
            }
        except Exception as e:
            logger.error(f"Error calculating market liquidity: {e}")
            result["market_liquidity_error"] = str(e)

        logger.info(f"Liquidity Risk Assessment Complete: {result}")
        return result

    def _calculate_lcr(self, financial_data: Dict[str, Any]) -> float:
        """Calculates Liquidity Coverage Ratio (LCR) approximation."""
        # LCR = HQLA / Total Net Cash Outflows over 30 days

        cash = financial_data.get("cash_and_equivalents", 0)
        govt_bonds = financial_data.get("government_bonds", 0)
        corp_bonds = financial_data.get("corporate_bonds_high_grade", 0)

        # HQLA proxy: Cash + 100% Govt + 85% Corp
        hqla = cash + (govt_bonds * 1.0) + (corp_bonds * 0.85)

        cl = financial_data.get("current_liabilities", 0)
        net_outflows = financial_data.get("net_cash_outflows_30d")

        if net_outflows is None or net_outflows == 0:
            # Proxy: 20% of Current Liabilities assume due in 30 days
            if cl > 0:
                net_outflows = cl * 0.20
            else:
                net_outflows = 1.0 # Avoid div by zero

        if net_outflows <= 0:
            return 10.0 # High liquidity implied if no outflows

        return hqla / net_outflows

    def _calculate_impact_cost(self, market_data: Dict[str, Any], trade_size: float) -> float:
        """Calculates estimated market impact cost (percentage) for a given trade size."""
        spread = market_data.get("bid_ask_spread", 0)
        price = market_data.get("price", 100)
        avg_vol = market_data.get("avg_daily_volume", 1000000)
        volatility = market_data.get("volatility", 0.02) # Daily vol

        if avg_vol <= 0: return 1.0 # 100% impact (illiquid)
        if price <= 0: return 0.0

        # Impact Model: Half-Spread + Temporary Impact (Square Root Law)
        # Cost = (Spread/2) + c * sigma * sqrt(Size / Volume)
        # Assuming c approx 0.5 to 1.0. Let's use 1.0 conservative.

        half_spread_pct = (spread / 2) / price

        # Market Impact (Price Move)
        # If trade size > avg_vol, impact is massive.
        ratio = trade_size / avg_vol
        market_impact_pct = volatility * np.sqrt(ratio)

        return float(half_spread_pct + market_impact_pct)
