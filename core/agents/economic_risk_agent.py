from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class EconomicRiskAgent(AgentBase):
    """
    Agent responsible for assessing Economic Risk (Macro).
    Evaluates recession risk, stagflation, Phillips Curve deviations, and Misery Index.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess economic risk.

        Args:
            macro_data: Dictionary containing:
                - gdp_growth (float, %)
                - inflation_rate (float, %)
                - unemployment_rate (float, %)
                - yield_curve_inversion (bool)

        Returns:
            Dict containing economic risk assessment.
        """
        logger.info("Starting Economic Risk Assessment...")

        gdp = macro_data.get("gdp_growth", 0.02)
        inflation = macro_data.get("inflation_rate", 0.02)
        unemp = macro_data.get("unemployment_rate", 0.04)
        inverted_yield = macro_data.get("yield_curve_inversion", False)

        risk_score, scenario, misery_index = self._analyze_macro_scenario(gdp, inflation, unemp, inverted_yield)

        # 4. Sahm Rule (Recession Indicator)
        unemp_history = macro_data.get("unemployment_history_12m", [])
        sahm_signal, sahm_val = self._apply_sahm_rule(unemp, unemp_history)
        if sahm_signal:
            risk_score += 25
            if "Recession" not in scenario:
                 scenario += " (Sahm Rule Triggered)"

        # 5. Taylor Rule (Monetary Policy Gap)
        fed_funds = macro_data.get("fed_funds_rate", 0.05)
        target_inflation = macro_data.get("target_inflation", 0.02)
        r_star = macro_data.get("natural_interest_rate", 0.005) # 0.5% real neutral rate

        taylor_rate = self._calculate_taylor_rule(inflation, gdp, target_inflation, r_star)
        policy_gap = fed_funds - taylor_rate

        # If policy is too loose (gap negative) -> Inflation Risk
        # If policy is too tight (gap positive) -> Recession Risk
        if abs(policy_gap) > 0.02:
            risk_score += 10 # Policy error risk

        risk_score = min(100.0, risk_score)
        level = "High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low"

        result = {
            "economic_risk_score": float(risk_score),
            "risk_level": level,
            "macro_scenario": scenario,
            "metrics": {
                "misery_index": float(misery_index),
                "misery_classification": self._classify_misery(misery_index),
                "yield_curve_signal": "Warning" if inverted_yield else "Normal",
                "sahm_rule_indicator": float(sahm_val),
                "sahm_signal": sahm_signal
            },
            "monetary_policy_analysis": {
                "actual_rate": float(fed_funds),
                "taylor_rule_rate": float(taylor_rate),
                "policy_gap": float(policy_gap),
                "interpretation": "Too Tight" if policy_gap > 0.01 else "Too Loose" if policy_gap < -0.01 else "Neutral"
            },
            "input_data": {
                "gdp_growth": gdp,
                "inflation_rate": inflation,
                "unemployment_rate": unemp
            }
        }

        logger.info(f"Economic Risk Assessment Complete: {result}")
        return result

    def _analyze_macro_scenario(self, gdp, inflation, unemployment, yield_inverted):
        """Analyzes the macro scenario and calculates a risk score."""
        misery_index = inflation + unemployment

        score = 0
        scenario = "Normal Expansion"

        # 1. Misery Index Component (Inflation + Unemployment)
        # Range logic:
        # < 6%: Great (0)
        # 6-9%: Normal (10)
        # 9-12%: Concerning (30)
        # 12-16%: High (50)
        # > 16%: Crisis (70)

        if misery_index > 0.16: score += 70
        elif misery_index > 0.12: score += 50
        elif misery_index > 0.09: score += 30
        elif misery_index > 0.06: score += 10

        # 2. GDP Growth Component & Scenario Labeling
        if gdp < -0.02:
            score += 30
            scenario = "Deep Recession"
        elif gdp < 0.00:
            score += 20
            scenario = "Recession"
        elif gdp < 0.015:
            # Low Growth
            if inflation > 0.04:
                scenario = "Stagflation"
                score += 20 # Additional penalty for stagflation
            else:
                scenario = "Stagnation"
                score += 10
        elif gdp > 0.04:
            # High Growth
            if inflation > 0.04:
                scenario = "Overheating"
                score += 10
            else:
                scenario = "Boom"

        # 3. Yield Curve Component
        if yield_inverted:
            score += 20
            if scenario == "Normal Expansion":
                scenario = "Late Cycle (Yield Inversion)"

        return min(100, score), scenario, misery_index

    def _classify_misery(self, misery_index):
        if misery_index < 0.06: return "Low"
        if misery_index < 0.10: return "Moderate"
        if misery_index < 0.15: return "High"
        return "Severe"

    def _apply_sahm_rule(self, current_unemp: float, unemp_history: list[float]) -> tuple[bool, float]:
        """
        Sahm Rule: Recession signaled when 3-month moving average of national unemployment rate
        rises by 0.50 percentage points or more relative to its low during the previous 12 months.
        """
        if not unemp_history or len(unemp_history) < 12:
            return False, 0.0

        # Combine current with history to get full series
        # Assume history is chronological [t-12, ..., t-1]
        full_series = unemp_history + [current_unemp]

        # Get last 3 months
        recent_3m = full_series[-3:]
        current_3m_avg = sum(recent_3m) / len(recent_3m)

        # Get lowest 3m average in previous 12 months
        # We need rolling 3m averages for the last 12 months
        rolling_avgs = []
        for i in range(len(full_series) - 2):
            window = full_series[i:i+3]
            rolling_avgs.append(sum(window)/3.0)

        if not rolling_avgs: return False, 0.0

        # Current is the last one
        curr_val = rolling_avgs[-1]

        # Previous ones (excluding current)
        prev_vals = rolling_avgs[:-1]
        if not prev_vals: prev_vals = [curr_val]

        min_val = min(prev_vals)

        diff = curr_val - min_val

        return (diff >= 0.005), float(diff)

    def _calculate_taylor_rule(self, inflation: float, gdp_growth: float, target_inflation: float = 0.02, r_star: float = 0.005) -> float:
        """
        Standard Taylor Rule: i = r* + pi + 0.5(pi - pi*) + 0.5(y - y*)
        Here we use GDP growth deviation from trend (approx 2%) as output gap proxy.
        """
        # Assumptions
        trend_growth = 0.02
        output_gap = gdp_growth - trend_growth # Rough proxy

        # Taylor Rule Formula
        # i = r_star + inflation + 0.5 * (inflation - target_inflation) + 0.5 * output_gap

        i_taylor = r_star + inflation + 0.5 * (inflation - target_inflation) + 0.5 * output_gap

        return float(i_taylor)
