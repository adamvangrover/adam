from core.agents.agent_base import AgentBase
from core.infrastructure.semantic_cache import SemanticCache
from core.system.state_manager import StateManager
from typing import Dict, Any, List, Optional
import logging
import math
from pydantic import BaseModel, Field

class Scenario(BaseModel):
    name: str
    description: str
    rate_shock_bps: float = 0.0
    gdp_shock_pct: float = 0.0
    revenue_shock_pct: float = 0.0
    cost_shock_pct: float = 0.0
    vix_level: float = 20.0

class SensitivityResult(BaseModel):
    scenario_name: str
    implied_pd: float
    rating_downgrade: bool
    projected_debt_to_equity: float
    projected_interest_coverage: float

class BlackSwanAgent(AgentBase):
    """
    Counterfactual 'Black Swan' Engine.
    Autonomously generates stress scenarios and calculates 'Probability of Default' sensitivity.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.cache = SemanticCache()
        self.state_manager = StateManager()

    DEFAULT_SCENARIOS = [
        Scenario(
            name="Rate Shock",
            description="Interest rates rise by 200bps.",
            rate_shock_bps=200.0,
            vix_level=25.0
        ),
        Scenario(
            name="Recession",
            description="GDP contracts by 2%, Revenue falls by 15%.",
            gdp_shock_pct=-2.0,
            revenue_shock_pct=-15.0,
            vix_level=35.0
        ),
        Scenario(
            name="Stagflation",
            description="Rates up 150bps, Costs up 10%, Revenue flat.",
            rate_shock_bps=150.0,
            cost_shock_pct=10.0,
            vix_level=30.0
        ),
        Scenario(
            name="Commodity Crash",
            description="Revenue falls 30% due to price shock.",
            revenue_shock_pct=-30.0,
            vix_level=40.0
        )
    ]

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the Black Swan analysis.
        Expected kwargs:
            financial_data: Dict containing 'key_ratios', 'total_debt', 'ebitda', 'interest_expense', 'revenue'
            scenarios: Optional List[Scenario]
        """
        logging.info(f"BlackSwanAgent execution started.")

        financial_data = kwargs.get('financial_data', {})
        scenarios_input = kwargs.get('scenarios')

        # Snapshot State (Rewind Button)
        snap_id = self.state_manager.save_snapshot(
            agent_id=self.config.get('name', 'BlackSwanAgent'),
            step_description="Pre-Analysis Execution",
            memory=getattr(self, 'memory', {}),
            context=kwargs
        )
        logging.info(f"State saved. Snapshot ID: {snap_id}")

        # Check Cache
        input_hash = SemanticCache.compute_data_hash({
            "financials": financial_data,
            "scenarios": str(scenarios_input)
        })
        cached = self.cache.get("BlackSwanAnalysis", input_hash, "v1_heuristic")
        if cached:
            logging.info("Cache Hit: Returning cached sensitivity analysis.")
            cached["_cache_hit"] = True
            return cached

        # Parse Scenarios
        scenarios = self.DEFAULT_SCENARIOS
        if scenarios_input and isinstance(scenarios_input, list):
            try:
                # Attempt to convert dicts to Scenario objects if they aren't already
                parsed_scenarios = []
                for s in scenarios_input:
                    if isinstance(s, Scenario):
                        parsed_scenarios.append(s)
                    elif isinstance(s, dict):
                        parsed_scenarios.append(Scenario(**s))

                if parsed_scenarios:
                    scenarios = parsed_scenarios
            except Exception as e:
                logging.warning(f"Failed to parse custom scenarios: {e}. Using defaults.")

        results = []

        base_pd = self._calculate_pd(financial_data)
        logging.info(f"Base PD: {base_pd:.2%}")

        for scenario in scenarios:
            stressed_financials = self._apply_stress(financial_data, scenario)
            stressed_pd = self._calculate_pd(stressed_financials)

            # Check for downgrade triggers
            # Simplified logic: if PD doubles or crosses 5%
            downgrade = (stressed_pd > 0.05) or (stressed_pd > base_pd * 2)

            results.append(SensitivityResult(
                scenario_name=scenario.name,
                implied_pd=stressed_pd,
                rating_downgrade=downgrade,
                projected_debt_to_equity=stressed_financials.get('key_ratios', {}).get('debt_to_equity_ratio', 0.0),
                projected_interest_coverage=stressed_financials.get('key_ratios', {}).get('interest_coverage_ratio', 0.0)
            ))

        # Generate Sensitivity Table (Markdown)
        md_table = "| Scenario | Implied PD | Debt/Equity | ICR | Downgrade Trigger? |\n"
        md_table += "|---|---|---|---|---|\n"
        md_table += f"| **Baseline** | {base_pd:.2%} | {financial_data.get('key_ratios', {}).get('debt_to_equity_ratio', 0.0):.2f}x | {financial_data.get('key_ratios', {}).get('interest_coverage_ratio', 0.0):.2f}x | - |\n"

        for r in results:
             md_table += f"| {r.scenario_name} | {r.implied_pd:.2%} | {r.projected_debt_to_equity:.2f}x | {r.projected_interest_coverage:.2f}x | {'**YES**' if r.rating_downgrade else 'No'} |\n"

        output = {
            "base_pd": base_pd,
            "sensitivity_analysis": [r.model_dump() for r in results],
            "sensitivity_table_markdown": md_table,
            "recommendation": "Review scenarios with Rating Downgrade flag." if any(r.rating_downgrade for r in results) else "Resilient to standard stress."
        }

        # Cache Output
        self.cache.set("BlackSwanAnalysis", input_hash, "v1_heuristic", output)

        return output

    def _apply_stress(self, financials: Dict[str, Any], scenario: Scenario) -> Dict[str, Any]:
        """
        Applies shocks to financial metrics to create a stressed dataset.
        """
        # Deep copy structure simplified
        stressed = financials.copy()
        ratios = stressed.get('key_ratios', {}).copy()
        stressed['key_ratios'] = ratios

        # Base metrics (defaults if missing to avoid crashes, though simplistic)
        revenue = financials.get('revenue', 1000)
        ebitda = financials.get('ebitda', 200)
        total_debt = financials.get('total_debt', 500)
        interest_expense = financials.get('interest_expense', 50)
        equity = financials.get('total_equity', 500)

        # Apply Shocks
        new_revenue = revenue * (1 + scenario.revenue_shock_pct / 100.0)
        # Assuming EBITDA margin degrades with revenue drop or cost shock
        # Simplified: EBITDA drops more than revenue (operating leverage)
        # And cost shock reduces EBITDA directly

        # Estimate Costs
        costs = revenue - ebitda
        new_costs = costs * (1 + scenario.cost_shock_pct / 100.0)

        new_ebitda = new_revenue - new_costs

        # Rate shock impact on interest (assuming 50% floating rate for simplicity)
        floating_debt_share = 0.5
        interest_increase = (total_debt * floating_debt_share) * (scenario.rate_shock_bps / 10000.0)
        new_interest_expense = interest_expense + interest_increase

        # Recalculate Ratios
        new_icr = new_ebitda / new_interest_expense if new_interest_expense > 0 else 999.0

        # Assume loss reduces equity
        # Net Income proxy ~ EBITDA - Interest - Tax (ignore tax for stress)
        old_ni = ebitda - interest_expense
        new_ni = new_ebitda - new_interest_expense
        delta_ni = new_ni - old_ni
        new_equity = equity + delta_ni # Impact on Retained Earnings

        new_de = total_debt / new_equity if new_equity > 0 else 999.0

        ratios['interest_coverage_ratio'] = new_icr
        ratios['debt_to_equity_ratio'] = new_de
        ratios['net_profit_margin'] = new_ni / new_revenue if new_revenue > 0 else -1.0

        stressed['revenue'] = new_revenue
        stressed['ebitda'] = new_ebitda
        stressed['total_equity'] = new_equity
        stressed['interest_expense'] = new_interest_expense

        return stressed

    def _calculate_pd(self, financials: Dict[str, Any]) -> float:
        """
        Calculates a synthetic Probability of Default based on Z-score like logic.
        This is a heuristic model for demonstration.
        """
        ratios = financials.get('key_ratios', {})
        icr = ratios.get('interest_coverage_ratio', 2.0)
        de = ratios.get('debt_to_equity_ratio', 1.0)
        profit_margin = ratios.get('net_profit_margin', 0.1)

        # Logistic Regression-ish components
        # Lower ICR -> Higher Risk
        score = 0.0
        if icr < 1.0: score += 2.0
        elif icr < 2.0: score += 1.0

        # Higher D/E -> Higher Risk
        if de > 4.0: score += 2.0
        elif de > 2.0: score += 1.0

        # Negative Profit -> Risk
        if profit_margin < 0: score += 1.5

        # Sigmoid function to map score to 0-1 (PD)
        # Shift so score 0 gives low PD (e.g. 1%)
        # Score 5 gives high PD

        # PD = 1 / (1 + exp(-(x - offset)))
        # Adjusting coefficients for "realistic" looking values
        logit = -3.0 + score
        pd = 1 / (1 + math.exp(-logit))

        return pd

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "BlackSwanAgent",
            "description": "Generates stress scenarios and sensitivity analysis.",
            "skills": [
                {
                    "name": "run_stress_test",
                    "description": "Runs stress test scenarios on financial data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "financial_data": {"type": "object", "description": "Financial metrics including total_debt, ebitda, interest_expense, etc."}
                        },
                        "required": ["financial_data"]
                    }
                }
            ]
        }
