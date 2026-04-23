import random
from typing import Dict, Any

class MockLLMGenerator:
    """
    Simulates a high-fidelity LLM response for financial crisis scenarios.
    Generates "High Conviction" insights based on asset attributes.
    """

    def __init__(self):
        self.sectors = ["Tech", "Energy", "Healthcare", "Financials", "Industrials"]
        self.scenarios = [
            "Liquidity Crunch",
            "Rate Hike Shock",
            "Supply Chain Disruption",
            "Regulatory Crackdown",
            "Cyber Event"
        ]
        self.templates = {
            "Credit Risk": [
                "Covenant headroom for {asset} has compressed to {headroom}x, signaling immediate default risk.",
                "{asset}'s EBITDA adjustments appear aggressive; true leverage is likely {leverage}x higher than reported.",
                "Cross-default provisions in {asset}'s documentation triggered by {sector} volatility.",
                "Rapid deterioration in {asset}'s interest coverage ratio (ICR) to {icr}x requires intervention.",
                "Structural subordination issues identified in {asset}'s capital stack."
            ],
            "Simulation": [
                "Under the '{scenario}' scenario, {asset} faces a {drawdown}% valuation haircut.",
                "Stress testing reveals {asset} has insufficient liquidity to survive a T+30 {scenario}.",
                "Kinetic simulation suggests {asset} correlates 0.85 with {sector} downside in a {scenario}.",
                "{asset} demonstrates resilience in a {scenario}, functioning as a portfolio hedge.",
                "Immediate capital injection required for {asset} to withstand the modeled {scenario}."
            ],
            "FIBO Extraction": [
                "Mapped {asset} to fibo-fbc-fi-fi:Loan with high confidence (0.98).",
                "Identified {asset} as fibo-be-le-lei:LegalEntity within the syndicated structure.",
                "Linked {asset} collateral to fibo-loan-ln-ln:Collateral with 0.92 certainty.",
                "Resolved {asset} entity relationships via knowledge graph traversal.",
                "Confirmed {asset} status as fibo-fbc-da-dbt:Debtor."
            ]
        }

    def generate_insight(self, asset: Dict[str, Any], prompt_category: str) -> str:
        """
        Generates a synthetic insight.
        """
        asset_name = asset.get("id", "Unknown Asset")

        # Determine template category
        category = "Credit Risk" # Default
        for key in self.templates.keys():
            if key.lower() in prompt_category.lower():
                category = key
                break

        template = random.choice(self.templates[category])

        # Generate random metrics
        headroom = round(random.uniform(0.1, 1.5), 2)
        leverage = round(random.uniform(0.5, 2.0), 1)
        icr = round(random.uniform(1.0, 2.5), 1)
        drawdown = round(random.uniform(15, 45), 1)
        sector = random.choice(self.sectors)
        scenario = random.choice(self.scenarios)

        return template.format(
            asset=asset_name,
            headroom=headroom,
            leverage=leverage,
            icr=icr,
            drawdown=drawdown,
            sector=sector,
            scenario=scenario
        )

    def score_response(self, asset: Dict[str, Any]) -> float:
        """
        Generates a risk/confidence score (0-100).
        """
        vol = asset.get("volatility", 0.2)
        # Higher vol -> Higher Risk Score
        base = 50
        risk_add = vol * 100
        noise = random.uniform(-10, 10)
        return min(100, max(0, round(base + risk_add + noise, 1)))
