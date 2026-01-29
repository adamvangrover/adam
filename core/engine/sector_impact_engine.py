import json
import os
import random
from typing import List, Dict, Any
from core.engine.contagion_engine import ContagionEngine

class SectorImpactEngine:
    """
    Simulates a "Consensus" by aggregating agent views on a portfolio
    against a live market backdrop.
    """

    def __init__(self):
        self.context_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'live_market_context.json')
        self.scenarios_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'scenarios.json')
        self.context = self._load_context()
        self.scenarios = self._load_scenarios()
        self.contagion = ContagionEngine()
        self.active_contagion_log = []

    def _load_context(self):
        try:
            with open(self.context_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {"themes": []}

    def _load_scenarios(self):
        try:
            with open(self.scenarios_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def analyze_portfolio(self, portfolio: List[Dict[str, Any]], scenario_id: str = None) -> List[Dict]:
        """
        Runs the simulation for the entire portfolio.
        """
        results = []
        themes = self.context.get('themes', [])

        # Determine Active Shocks (Scenario + Contagion)
        sector_shocks = {}
        self.active_contagion_log = []

        if scenario_id:
            scenario = next((s for s in self.scenarios if s['id'] == scenario_id), None)
            if scenario:
                initial_shocks = scenario.get('shocks', {})
                contagion_result = self.contagion.simulate_contagion(initial_shocks)
                sector_shocks = contagion_result['final_impacts']
                self.active_contagion_log = contagion_result['log']

        for asset in portfolio:
            # 1. Macro Agent View
            macro_view = self._agent_macro(asset, themes)

            # 2. Credit Agent View
            credit_view = self._agent_credit(asset)

            # 3. Geopolitical Agent View
            geo_view = self._agent_geopolitical(asset, themes)

            # Apply Scenario Shock to Agents
            # If the sector is under shock, Macro and Credit views worsen
            shock_val = sector_shocks.get(asset.get("sector"), 0.0)
            if shock_val < 0:
                # Shock is negative (distress), so Risk Score goes UP
                shock_penalty = abs(shock_val) * 100 * 0.5 # Scale roughly
                macro_view['score'] += shock_penalty
                credit_view['score'] += shock_penalty

                macro_view['insight'] += f" [SCENARIO IMPACT: {shock_val}]"

            # Consensus Synthesis
            consensus_score = (macro_view['score'] + credit_view['score'] + geo_view['score']) / 3
            consensus_score = min(100.0, consensus_score) # Cap at 100

            results.append({
                "asset": asset.get("name", asset.get("id")),
                "sector": asset.get("sector", "Unknown"),
                "macro_insight": macro_view['insight'],
                "credit_insight": credit_view['insight'],
                "geo_insight": geo_view['insight'],
                "consensus_score": round(consensus_score, 1),
                "risk_regime": self.context.get("macro_regime", "Neutral")
            })

        return results

    def _agent_macro(self, asset, themes):
        """Top-down view based on sector alignment with themes."""
        sector = asset.get("sector")
        relevant_theme = next((t for t in themes if sector in t.get("impact_sectors", [])), None)

        if relevant_theme:
            return {
                "score": 85.0 if relevant_theme['severity'] > 0.7 else 60.0,
                "insight": f"Sector exposed to '{relevant_theme['name']}'. Macro headwinds intensify."
            }
        return {
            "score": 40.0,
            "insight": "Sector effectively neutral to current macro regime."
        }

    def _agent_credit(self, asset):
        """Bottom-up view based on leverage/rating."""
        lev = asset.get("leverage", 4.0)
        rating = asset.get("rating", "B")

        if lev > 5.5 or "CCC" in rating:
            return {
                "score": 90.0,
                "insight": f"Critical leverage ({lev}x) creates refinance cliff risk."
            }
        elif lev > 4.5:
            return {
                "score": 65.0,
                "insight": "Moderate leverage; watch ICR compression."
            }
        return {
            "score": 25.0,
            "insight": "Strong balance sheet resilience."
        }

    def _agent_geopolitical(self, asset, themes):
        """External view."""
        # Simple heuristic: Energy/Tech are high geo risk
        sector = asset.get("sector")
        if sector in ["Energy", "Technology", "Industrials"]:
            return {
                "score": 75.0,
                "insight": "Supply chain fragmentation risk detected in tier-2 suppliers."
            }
        return {
            "score": 30.0,
            "insight": "Domestic focus insulates against trade tensions."
        }
