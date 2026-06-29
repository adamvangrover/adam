from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import numpy as np
import json
import hashlib
from datetime import datetime
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentOutput
from src.pdil.models import ProvenanceHeader
from src.pdil.middleware import JsonLogicGovernanceGatekeeper

logger = logging.getLogger(__name__)

class LiquidityRiskAgent(AgentBase):
    """
    Agent responsible for assessing Liquidity Risk.
    Calculates Financial Liquidity (Current/Quick Ratio, LCR)
    and Market Liquidity (Spread, Impact Cost).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # Define a jsonLogic rule ensuring positive current liabilities to avoid div by zero gracefully
        self.logic_rules = {
            ">": [{"var": "current_liabilities"}, 0]
        }
        self.gatekeeper = JsonLogicGovernanceGatekeeper(self.logic_rules)

    async def execute(self, data: Dict[str, Any]) -> AgentOutput:
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

        # Additive Governance: Evaluate financial data payload through jsonLogic Gatekeeper
        mock_input_validation_payload = {
            "provenance_trace": {
                "git_commit_hash": "current",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "content_hash": hashlib.sha256(json.dumps(financial_data, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
                "jsonLogic_version": "v2.0",
                "confidence_score": 1.0,
                "derivation_path": "input",
                "source_data_object": "input"
            },
            "data": financial_data
        }

        observed_drift = False
        try:
            self.gatekeeper.validate_inference(mock_input_validation_payload)
        except Exception as e:
            logger.warning(f"Financial data failed jsonLogic governance constraints: {e}")
            observed_drift = True
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

            # Cash Conversion Cycle (CCC)
            ccc = self._calculate_ccc(financial_data)

            # NSFR Approximation
            nsfr = self._calculate_nsfr(financial_data)

            result["financial_liquidity"] = {
                "current_ratio": float(current_ratio),
                "quick_ratio": float(quick_ratio),
                "liquidity_coverage_ratio_approx": float(lcr),
                "cash_conversion_cycle_days": ccc,
                "net_stable_funding_ratio_approx": float(nsfr),
                "assessment": "Good" if lcr > 1.0 and (ccc is None or ccc < 90) else "Concern" if lcr < 0.8 else "Adequate"
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

            # Spread Volatility (if history available)
            spread_vol = self._calculate_spread_volatility(market_data)

            result["market_liquidity"] = {
                "avg_daily_volume": float(avg_volume),
                "bid_ask_spread_pct": float(spread_pct),
                "simulated_impact_cost_pct": float(impact_cost),
                "simulated_trade_size": float(trade_size),
                "bid_ask_spread_volatility": float(spread_vol),
                "assessment": "Liquid" if impact_cost < 0.01 else "Illiquid" if impact_cost > 0.05 else "Moderate"
            }
        except Exception as e:
            logger.error(f"Error calculating market liquidity: {e}")
            result["market_liquidity_error"] = str(e)

        logger.info(f"Liquidity Risk Assessment Complete: {result}")

        # Dynamic Provenance Propagation
        payload_json = json.dumps(result, sort_keys=True, separators=(',', ':')).encode('utf-8')
        content_hash = hashlib.sha256(payload_json).hexdigest()

        provenance = ProvenanceHeader(
            git_commit_hash="current",
            timestamp=datetime.utcnow().isoformat() + "Z",
            content_hash=content_hash,
            jsonLogic_version="v2.0",
            confidence_score=0.9,
            derivation_path="LiquidityRiskAgent -> Financial & Market Liquidity",
            source_data_object="internal_calculation"
        )

        return AgentOutput(
            answer="Liquidity risk assessment completed successfully.",
            sources=["internal_calculation"],
            confidence=0.9,
            metadata={"financial_liquidity_assessment": result.get("financial_liquidity", {}).get("assessment")},
            provenance_trace=provenance,
            data=result,
            observed_drift=observed_drift
        )

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

    def _calculate_ccc(self, financial_data: Dict[str, Any]) -> Optional[float]:
        """Calculates Cash Conversion Cycle (CCC) = DSO + DIO - DPO."""
        sales = financial_data.get("sales", 0)
        cogs = financial_data.get("cogs", 0)
        receivables = financial_data.get("receivables", 0)
        inventory = financial_data.get("inventory", 0)
        payables = financial_data.get("payables", 0)

        if sales == 0 or cogs == 0:
            return None

        # Days Sales Outstanding (DSO)
        dso = (receivables / sales) * 365

        # Days Inventory Outstanding (DIO)
        dio = (inventory / cogs) * 365

        # Days Payable Outstanding (DPO)
        dpo = (payables / cogs) * 365 # Often uses COGS or Purch (approx COGS)

        return float(dso + dio - dpo)

    def _calculate_nsfr(self, financial_data: Dict[str, Any]) -> float:
        """Approximates Net Stable Funding Ratio (Available Stable Funding / Required Stable Funding)."""
        # ASF Factors (Simplified Basel III)
        # Capital + Liabilities > 1yr: 100%
        # Stable Deposits: 95% (Not usually broken out, assume 0 for generic)
        # Wholesale Funding < 1yr: 0% or 50%

        equity = financial_data.get("market_value_equity", 0) # Proxy for capital
        long_term_debt = financial_data.get("long_term_debt", 0)

        asf = equity + long_term_debt # Conservative proxy

        # RSF Factors
        # Assets that need funding
        # Cash: 0%
        # Govt Bonds: 5%
        # Corp Bonds: 15-50%
        # Loans < 1yr: 50%
        # Loans > 1yr / Physical Assets: 100%

        # We use Total Assets - Cash - Govt Bonds as a proxy for "Illiquid Assets" needing funding
        total_assets = financial_data.get("total_assets", 0)
        cash = financial_data.get("cash_and_equivalents", 0)
        govt_bonds = financial_data.get("government_bonds", 0)

        illiquid_assets = total_assets - cash - govt_bonds
        if illiquid_assets < 0: illiquid_assets = 0

        # Rough RSF: 100% of Illiquid + 0% of Liquid
        rsf = illiquid_assets

        if rsf == 0: return 2.0 # Very safe

        return asf / rsf

    def _calculate_spread_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculates volatility of the bid-ask spread if history is provided."""
        spread_history = market_data.get("spread_history", [])
        if not spread_history or len(spread_history) < 2:
            return 0.0

        return float(np.std(spread_history))
