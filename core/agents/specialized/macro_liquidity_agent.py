from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
from pydantic import BaseModel, Field
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None

from core.agents.agent_base import AgentBase
from core.data_sources.data_fetcher import DataFetcher

logger = logging.getLogger("MacroLiquidityAgent")

# --- Pydantic Models ---

class MacroLiquidityInput(BaseModel):
    """Input model for Macro Liquidity Analysis."""
    force_refresh: bool = Field(False, description="Whether to bypass cache and force fresh data fetch.")
    symbol_overrides: Optional[Dict[str, str]] = Field(None, description="Override default tickers (e.g., {'tnx': '^TNX'})")

class MacroLiquidityOutput(BaseModel):
    """Output model for Macro Liquidity Analysis."""
    liquidity_score: float = Field(..., ge=0.0, le=100.0, description="0 (Abundant) to 100 (Crisis) Stress Index.")
    regime: str = Field(..., description="Current liquidity regime (e.g., 'Contractionary', 'Neutral', 'Expansionary').")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the assessment based on data availability.")
    components: Dict[str, Any] = Field(..., description="Breakdown of key indicators (yields, spreads, etc.).")
    anomalies: List[str] = Field(default_factory=list, description="List of detected market anomalies.")
    timestamp: str = Field(..., description="ISO format timestamp of the analysis.")
    reasoning_trace: str = Field(..., description="Human-readable explanation of the score.")

class MacroLiquidityAgent(AgentBase):
    """
    Agent responsible for assessing global macro liquidity conditions by analyzing
    bond yields, credit spreads, currency strength, and commodity proxies.

    It calculates a 'Liquidity Stress Index' that serves as a fundamental input
    for Risk Agents and Portfolio Managers.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.data_fetcher = DataFetcher()
        self.tickers = {
            "TNX": "^TNX",      # 10Y Treasury Yield
            "IRX": "^IRX",      # 13W Treasury Bill
            "HYG": "HYG",       # High Yield Bond ETF
            "LQD": "LQD",       # Investment Grade Bond ETF
            "DXY": "DX-Y.NYB",  # US Dollar Index (Yahoo ticker)
            "GOLD": "GC=F"      # Gold Futures
        }
        # Allow config to override tickers
        if config.get("tickers"):
            self.tickers.update(config["tickers"])

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the macro liquidity analysis.
        """
        logger.info("MacroLiquidityAgent execution started.")

        # Parse inputs
        try:
            input_data = MacroLiquidityInput(**kwargs)
        except Exception as e:
            logger.warning(f"Invalid input, using defaults: {e}")
            input_data = MacroLiquidityInput()

        if input_data.symbol_overrides:
            self.tickers.update(input_data.symbol_overrides)

        # 1. Fetch Data
        data, fetch_confidence = await self._fetch_macro_data()

        # 2. Calculate Indicators
        indicators = self._calculate_indicators(data)

        # 3. Compute Stress Index & Regime
        score, regime, trace = self._compute_stress_index(indicators)

        # 4. Identify Anomalies
        anomalies = self._detect_anomalies(indicators)

        # 5. Final Confidence Adjustment
        # Confidence is product of data availability and internal logic consistency (assumed 1.0 for logic)
        final_confidence = fetch_confidence

        # Construct Output
        output = MacroLiquidityOutput(
            liquidity_score=score,
            regime=regime,
            confidence_score=final_confidence,
            components=indicators,
            anomalies=anomalies,
            timestamp=pd.Timestamp.now().isoformat() if pd else datetime.now().isoformat(),
            reasoning_trace=trace
        )

        logger.info(f"Macro Liquidity Analysis Complete: Score={score}, Regime={regime}")
        return output.model_dump()

    async def _fetch_macro_data(self) -> Tuple[Dict[str, Any], float]:
        """
        Fetches necessary data points. Returns data dict and confidence score (0.0 - 1.0).
        """
        data = {}
        success_count = 0
        total_required = len(self.tickers)

        # Helper to run blocking yfinance calls in executor
        loop = asyncio.get_running_loop()

        async def fetch_ticker(key, symbol):
            try:
                # Use DataFetcher for standard market data structure
                # We use fetch_market_data or fetch_realtime_snapshot
                # DataFetcher methods are blocking, so we run them in executor
                result = await loop.run_in_executor(None, self.data_fetcher.fetch_realtime_snapshot, symbol)

                # Check if result is valid
                if result and result.get("last_price"):
                    return key, result

                # Fallback to standard market data
                result = await loop.run_in_executor(None, self.data_fetcher.fetch_market_data, symbol)
                if result and result.get("current_price"):
                    return key, result

                return key, None
            except Exception as e:
                logger.error(f"Error fetching {key} ({symbol}): {e}")
                return key, None

        tasks = [fetch_ticker(k, v) for k, v in self.tickers.items()]
        results = await asyncio.gather(*tasks)

        for key, result in results:
            if result:
                data[key] = result
                # Check if simulated
                if not result.get("simulated", False):
                    success_count += 1
                else:
                    # Simulated data counts as partial success (0.5) for confidence
                    success_count += 0.5
            else:
                logger.warning(f"Missing data for {key}")

        confidence = success_count / total_required if total_required > 0 else 0.0
        return data, confidence

    def _calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derives secondary indicators (spreads, curves).
        """
        indicators = {}

        # Safe getter
        def get_price(key):
            item = data.get(key)
            if not item: return None
            return item.get("last_price") or item.get("current_price")

        tnx = get_price("TNX")
        irx = get_price("IRX")
        hyg = get_price("HYG")
        lqd = get_price("LQD")
        dxy = get_price("DXY")
        gold = get_price("GOLD")

        indicators["tnx_yield"] = tnx
        indicators["irx_yield"] = irx
        indicators["dxy_level"] = dxy
        indicators["gold_price"] = gold

        # 1. Yield Curve (10Y - 3M)
        if tnx is not None and irx is not None:
            indicators["yield_curve_slope"] = tnx - irx
        else:
            indicators["yield_curve_slope"] = None

        # 2. Credit Spread Proxy (LQD/HYG Ratio or simple diff?)
        # HYG (Junk) should be riskier than LQD (Inv Grade).
        # We can look at the RATIO of Prices.
        # If HYG/LQD drops, High Yield is underperforming -> Stress.
        if hyg is not None and lqd is not None and lqd != 0:
            indicators["credit_ratio"] = hyg / lqd
        else:
            indicators["credit_ratio"] = None

        return indicators

    def _compute_stress_index(self, indicators: Dict[str, Any]) -> Tuple[float, str, str]:
        """
        Calculates the Liquidity Stress Index (0-100).
        """
        score = 50.0 # Start neutral
        reasons = []

        # Logic is heuristic-based for now.
        # In a real system, this would use Z-Scores relative to moving averages.

        # 1. Yield Curve Inversion
        slope = indicators.get("yield_curve_slope")
        if slope is not None:
            if slope < 0:
                score += 20
                reasons.append(f"Yield Curve Inverted ({slope:.2f}%)")
            elif slope < 0.5:
                score += 10
                reasons.append("Yield Curve Flattening")
            else:
                score -= 10
                reasons.append("Yield Curve Steep/Normal")

        # 2. Credit Stress
        credit_ratio = indicators.get("credit_ratio")
        # Heuristic: Recent norms for HYG/LQD is around 0.70 - 0.75
        # If it drops below 0.68, stress is high.
        if credit_ratio is not None:
            if credit_ratio < 0.68:
                score += 20
                reasons.append(f"Credit Spreads Widening (Ratio: {credit_ratio:.3f})")
            elif credit_ratio > 0.72:
                score -= 10
                reasons.append("Credit Spreads Healthy")

        # 3. Dollar Strength
        dxy = indicators.get("dxy_level")
        # DXY > 105 is usually restrictive for global liquidity
        if dxy is not None:
            if dxy > 105:
                score += 15
                reasons.append(f"Strong Dollar (DXY: {dxy:.2f})")
            elif dxy < 100:
                score -= 5
                reasons.append("Weak Dollar (Supportive)")

        # 4. Yield Levels
        tnx = indicators.get("tnx_yield")
        # TNX > 4.5% starts to break things
        if tnx is not None:
            if tnx > 4.5:
                score += 15
                reasons.append(f"High Rates (10Y: {tnx:.2f}%)")
            elif tnx < 3.5:
                score -= 10
                reasons.append("Rates Moderate")

        # Clamp Score
        score = max(0.0, min(100.0, score))

        # Determine Regime
        if score > 75:
            regime = "CRISIS / CONTRACTION"
        elif score > 55:
            regime = "TIGHTENING"
        elif score > 40:
            regime = "NEUTRAL"
        else:
            regime = "EXPANSIONARY"

        trace = "; ".join(reasons)
        return score, regime, trace

    def _detect_anomalies(self, indicators: Dict[str, Any]) -> List[str]:
        """
        Detects specific contradictions (e.g., Gold Up + Dollar Up).
        """
        anomalies = []

        dxy = indicators.get("dxy_level")
        gold = indicators.get("gold_price")
        tnx = indicators.get("tnx_yield")

        # Anomaly 1: Gold & Dollar Rising Together (Fear Bid)
        # We need change data for this, but using levels for heuristic check
        # Ideally this would look at daily % change.
        # For this version, we'll skip change-based anomalies without history.

        # Anomaly 2: Yields Up, Gold Up (Inflation expectations unanchoring?)
        if tnx and gold and tnx > 4.5 and gold > 2100:
             anomalies.append("High Yields + High Gold: Inflation Fear?")

        return anomalies

if __name__ == "__main__":
    # Test Harness
    logging.basicConfig(level=logging.INFO)
    async def main():
        agent = MacroLiquidityAgent({})
        result = await agent.execute()
        print(result)

    asyncio.run(main())
