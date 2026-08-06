import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

# Assuming these are available from your core and interfaces packages
from afos_core import Policy, Scenario
from afos_interfaces import ISimulationKernel

logger = logging.getLogger(__name__)

# ==================================================================
# SIMULATION KERNEL
# ==================================================================
class SimulationKernel(ISimulationKernel):
    """
    Concrete implementation of the Simulation Kernel.
    Executes macroeconomic shock scenarios and alternative policy sets 
    against portfolio data to quantify systemic risk and expected loss (VaR).
    """
    def __init__(self):
        self._active_simulations: int = 0
        # Mock portfolio database for standalone execution
        self._mock_portfolio_db: Dict[str, List[Dict[str, Any]]] = self._hydrate_mock_portfolios()

    async def initialize(self) -> None:
        """Boot sequence: Allocate memory arrays for high-velocity matrix operations."""
        logger.info("Initializing Simulation Kernel: Quantitative Engineering & Stress Testing engine online.")
        # In production: Initialize GPU clusters or Databricks agentic data stack connections
        await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        """Teardown sequence."""
        logger.info(f"Simulation Kernel shutting down. Active simulations halted: {self._active_simulations}")

    async def run_simulation(self, portfolio_id: str, policy: Policy, scenario: Scenario) -> Dict[str, Any]:
        """
        Replays a portfolio against hypothetical policies and macro shock factors.
        Calculates delta in covenant breaches, exposure at default, and risk migration.
        """
        self._active_simulations += 1
        sim_id = f"sim_{uuid.uuid4().hex[:8]}"
        logger.info(f"[SIMULATION {sim_id}] Initiating scenario '{scenario.description}' on portfolio '{portfolio_id}'")

        try:
            # 1. Fetch Baseline Data
            baseline_assets = self._mock_portfolio_db.get(portfolio_id)
            if not baseline_assets:
                raise ValueError(f"Portfolio {portfolio_id} not found in localized memory.")

            # 2. Extract Macro Shocks
            ebitda_shock = scenario.parameters.get("ebitda_compression_pct", 0.0)
            rate_shock_bps = scenario.parameters.get("interest_rate_shock_bps", 0.0)
            
            logger.debug(f"[SIMULATION {sim_id}] Applying Shocks -> EBITDA: {ebitda_shock*100}%, Rates: +{rate_shock_bps}bps")

            # 3. Apply Deterministic Shocks (Vectorized in production; iterative here)
            stressed_assets = await self._apply_macro_shocks(baseline_assets, ebitda_shock, rate_shock_bps)

            # 4. Evaluate Stressed Assets against Hypothetical Policy
            # In a full system, this would call the PolicyKernel. Here, we simulate the evaluation.
            results = await self._evaluate_stressed_portfolio(stressed_assets, policy)

            # 5. Compile Telemetry & VaR Output
            pre_shock_breaches = sum(1 for a in baseline_assets if a["leverage"] > 4.5) # Assuming baseline 4.5x limit
            post_shock_breaches = results["total_breaches"]

            report = {
                "simulation_id": sim_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scenario_applied": scenario.description,
                "policy_tested": policy.id,
                "telemetry": {
                    "assets_scanned": len(baseline_assets),
                    "baseline_covenant_breaches": pre_shock_breaches,
                    "stressed_covenant_breaches": post_shock_breaches,
                    "breach_delta": post_shock_breaches - pre_shock_breaches,
                    "stressed_exposure_at_risk": results["exposure_at_risk"]
                },
                "asset_level_results": results["asset_details"]
            }

            logger.info(f"[SIMULATION {sim_id}] Completed. Breach Delta: +{report['telemetry']['breach_delta']}")
            return report

        finally:
            self._active_simulations -= 1

    # ---------------------------------------------------------------
    # Internal Quantitative Mechanisms
    # ---------------------------------------------------------------
    async def _apply_macro_shocks(self, assets: List[Dict[str, Any]], ebitda_shock: float, rate_shock_bps: float) -> List[Dict[str, Any]]:
        """
        Applies mathematical transformations to asset financials to simulate market conditions.
        """
        await asyncio.sleep(0.2) # Simulate compute latency for matrix operations
        stressed = []
        
        for asset in assets:
            # Deep copy to avoid mutating the baseline
            s_asset = json.loads(json.dumps(asset))
            
            # Apply EBITDA compression
            original_ebitda = s_asset["financials"]["ebitda"]
            new_ebitda = original_ebitda * (1.0 + ebitda_shock)
            
            # Apply Rate Shock (converting bps to decimal, e.g., 150bps = 0.015)
            original_rate = s_asset["financials"]["base_rate"]
            new_rate = original_rate + (rate_shock_bps / 10000.0)
            
            # Recalculate derived metrics
            total_debt = s_asset["financials"]["total_debt"]
            new_leverage = total_debt / new_ebitda if new_ebitda > 0 else float('inf')
            
            new_interest_expense = total_debt * new_rate
            new_icr = new_ebitda / new_interest_expense if new_interest_expense > 0 else float('inf')
            
            # Update stressed asset record
            s_asset["financials"]["ebitda"] = new_ebitda
            s_asset["financials"]["base_rate"] = new_rate
            s_asset["financials"]["leverage"] = round(new_leverage, 2)
            s_asset["financials"]["interest_coverage"] = round(new_icr, 2)
            
            stressed.append(s_asset)
            
        return stressed

    async def _evaluate_stressed_portfolio(self, stressed_assets: List[Dict[str, Any]], policy: Policy) -> Dict[str, Any]:
        """
        Simulates the PolicyKernel mapping over the stressed portfolio to detect breaches.
        """
        # For standalone purposes, we extract a mock limit from the policy ruleset
        # Assuming ruleset looks like: {"<": [{"var": "financials.leverage"}, 4.0]}
        try:
            rules = json.loads(policy.rules)
            leverage_limit = rules.get("<", [{}, 4.5])[1] # Default to 4.5 if parse fails
        except:
            leverage_limit = 4.5

        total_breaches = 0
        exposure_at_risk = 0.0
        asset_details = []

        for asset in stressed_assets:
            is_breached = asset["financials"]["leverage"] >= leverage_limit
            if is_breached:
                total_breaches += 1
                exposure_at_risk += asset["exposure_amount"]
                
            asset_details.append({
                "entity_id": asset["entity_id"],
                "stressed_leverage": asset["financials"]["leverage"],
                "stressed_icr": asset["financials"]["interest_coverage"],
                "policy_breached": is_breached
            })

        return {
            "total_breaches": total_breaches,
            "exposure_at_risk": exposure_at_risk,
            "asset_details": asset_details
        }

    def _hydrate_mock_portfolios(self) -> Dict[str, List[Dict[str, Any]]]:
        """Provides deterministic baseline data for TMT & Leveraged Finance assets."""
        return {
            "port_tmt_levfin_01": [
                {
                    "entity_id": "org_saas_alpha",
                    "exposure_amount": 150000000.0,
                    "financials": {"ebitda": 50000000.0, "total_debt": 200000000.0, "base_rate": 0.05, "leverage": 4.0, "interest_coverage": 5.0}
                },
                {
                    "entity_id": "org_telecom_beta",
                    "exposure_amount": 300000000.0,
                    "financials": {"ebitda": 100000000.0, "total_debt": 420000000.0, "base_rate": 0.055, "leverage": 4.2, "interest_coverage": 4.3}
                },
                {
                    "entity_id": "org_media_gamma",
                    "exposure_amount": 75000000.0,
                    "financials": {"ebitda": 25000000.0, "total_debt": 110000000.0, "base_rate": 0.06, "leverage": 4.4, "interest_coverage": 3.7}
                }
            ]
        }

# ==================================================================
# EXECUTION / FUNCTIONAL TEST
# ==================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    async def main():
        kernel = SimulationKernel()
        await kernel.initialize()
        
        # 1. Define a strict Hypothetical Policy (Tightening Leverage limit to 4.25x)
        test_policy = Policy(
            id="pol_stress_test_v1",
            version="1.0",
            ruleset="jsonlogic",
            rules='{"<": [{"var": "financials.leverage"}, 4.25]}'
        )
        
        # 2. Define a Macroeconomic Scenario (Tech Contagion: -20% EBITDA, +150bps Rates)
        contagion_scenario = Scenario(
            id="scn_tech_contagion_2026",
            description="Severe TMT sector contraction with simultaneous cost-of-capital spike.",
            parameters={
                "ebitda_compression_pct": -0.20,  # 20% drop in EBITDA
                "interest_rate_shock_bps": 150.0  # 150 basis points increase
            }
        )
        
        # 3. Execute Simulation on the TMT Leveraged Finance Portfolio
        print("\n--- Running Quantitative Simulation ---")
        report = await kernel.run_simulation(
            portfolio_id="port_tmt_levfin_01",
            policy=test_policy,
            scenario=contagion_scenario
        )
        
        print("\n✅ Simulation Report Generated:")
        print(json.dumps(report, indent=2))

        await kernel.shutdown()

    # Run the event loop
    asyncio.run(main())
