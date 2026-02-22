from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class LiquidityRiskAgent(AgentBase):
    """
    Agent responsible for assessing Liquidity Risk.
    Calculates Financial Liquidity (Current Ratio) and Market Liquidity (Volume/Spread).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the liquidity risk assessment.

        Args:
            data: Dictionary containing:
                - financial_data: {current_assets, current_liabilities, inventory, ...}
                - market_data: {avg_daily_volume, bid_ask_spread, ...}

        Returns:
            Dict containing liquidity risk metrics.
        """
        logger.info("Starting Liquidity Risk Assessment...")

        financial_data = data.get("financial_data", {})
        market_data = data.get("market_data", {})

        result = {}

        # 1. Financial Liquidity (Balance Sheet)
        try:
            ca = financial_data.get("current_assets", 0)
            cl = financial_data.get("current_liabilities", 1)
            inv = financial_data.get("inventory", 0)

            if cl == 0: cl = 1

            current_ratio = ca / cl
            quick_ratio = (ca - inv) / cl

            result["financial_liquidity"] = {
                "current_ratio": float(current_ratio),
                "quick_ratio": float(quick_ratio),
                "assessment": "Good" if current_ratio > 1.5 else "Concern" if current_ratio < 1.0 else "Neutral"
            }
        except Exception as e:
            logger.error(f"Error calculating financial liquidity: {e}")
            result["financial_liquidity_error"] = str(e)

        # 2. Market Liquidity (Trading)
        try:
            avg_volume = market_data.get("avg_daily_volume", 0)
            bid_ask_spread = market_data.get("bid_ask_spread", 0)
            price = market_data.get("price", 100) # Default for spread calculation

            # Spread percentage
            spread_pct = (bid_ask_spread / price) if price > 0 else 0

            # Amihud Illiquidity Ratio Proxy (if return data available - omitted for simplicity)

            result["market_liquidity"] = {
                "avg_daily_volume": float(avg_volume),
                "bid_ask_spread_pct": float(spread_pct),
                "assessment": "Liquid" if avg_volume > 1000000 else "Illiquid" if avg_volume < 50000 else "Moderate"
            }
        except Exception as e:
            logger.error(f"Error calculating market liquidity: {e}")
            result["market_liquidity_error"] = str(e)

        logger.info(f"Liquidity Risk Assessment Complete: {result}")
        return result
