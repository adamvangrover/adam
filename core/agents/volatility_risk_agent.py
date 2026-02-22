from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import numpy as np
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class VolatilityRiskAgent(AgentBase):
    """
    Agent responsible for assessing Volatility Risk.
    Analyzes VIX, GARCH(1,1) forecasts, and Volatility Risk Premium (VRP).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess volatility risk.

        Args:
            vol_data: Dictionary containing:
                - vix_index (float)
                - implied_volatility (float, optional, e.g. 0.20 for 20%)
                - realized_volatility (float, optional, e.g. 0.15 for 15%)
                - current_return (float, optional, daily return e.g. -0.01)
                - previous_volatility (float, optional, annualized e.g. 0.18)

        Returns:
            Dict containing volatility risk assessment.
        """
        logger.info("Starting Volatility Risk Assessment...")

        vix = vol_data.get("vix_index", 20.0)
        # Normalize VIX to decimal if needed (usually passed as 20.0)

        iv = vol_data.get("implied_volatility", vix / 100.0)
        rv = vol_data.get("realized_volatility", iv * 0.9) # Default slightly lower

        risk_score = 0.0

        # 1. Volatility Regime
        if vix < 15:
            regime = "Low Volatility (Complacency?)"
            risk_score += 20
        elif vix < 25:
            regime = "Normal Volatility"
            risk_score += 40
        elif vix < 40:
            regime = "High Volatility (Stress)"
            risk_score += 70
        else:
            regime = "Extreme Volatility (Crisis)"
            risk_score += 95

        # 2. Volatility Risk Premium (VRP)
        # VRP = IV - RV
        vrp = iv - rv
        vrp_signal = "Normal"
        if vrp < -0.02:
            vrp_signal = "Danger: Volatility Spiking (IV < RV)"
            risk_score += 20
        elif vrp > 0.10:
            vrp_signal = "High Risk Premium (Fear)"
            risk_score += 10

        # 3. GARCH(1,1) Simplified Forecast (Daily Step)
        # We assume parameters for S&P 500 roughly
        # Long Run Vol assumption = 20% Annualized
        long_run_daily_vol = 0.20 / np.sqrt(252)
        target_daily_var = long_run_daily_vol ** 2

        alpha = 0.05
        beta = 0.90
        omega = target_daily_var * (1 - alpha - beta)

        current_ret = vol_data.get("current_return", 0.0)
        # previous_volatility assumed Annualized, convert to Daily
        prev_annual_vol = vol_data.get("previous_volatility", rv)
        prev_daily_vol = prev_annual_vol / np.sqrt(252)

        # GARCH Update
        next_daily_var = omega + alpha * (current_ret**2) + beta * (prev_daily_vol**2)
        next_daily_vol = np.sqrt(next_daily_var)

        # Convert back to Annualized
        garch_forecast_annual = next_daily_vol * np.sqrt(252)

        # Trend Analysis
        vol_trend = "Stable"
        if garch_forecast_annual > prev_annual_vol * 1.05:
            vol_trend = "Rising"
            risk_score += 5
        elif garch_forecast_annual < prev_annual_vol * 0.95:
            vol_trend = "Falling"

        risk_score = min(100.0, risk_score)
        level = "High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low"

        result = {
            "volatility_risk_score": float(risk_score),
            "risk_level": level,
            "regime": regime,
            "metrics": {
                "vix_index": float(vix),
                "implied_volatility": float(iv),
                "realized_volatility": float(rv),
                "volatility_risk_premium": float(vrp),
                "vrp_signal": vrp_signal
            },
            "garch_forecast": {
                "next_period_volatility_annualized": float(garch_forecast_annual),
                "trend": vol_trend,
                "model": "GARCH(1,1) Daily Step"
            }
        }

        logger.info(f"Volatility Risk Assessment Complete: {result}")
        return result
