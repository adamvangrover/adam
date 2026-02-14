import asyncio
import json
import os
import sys

# Ensure core is in path
sys.path.append(os.getcwd())

from core.agents.fraud_detection_agent import FraudDetectionAgent

async def main():
    print("Initializing Fraud Detection Agent...")
    agent = FraudDetectionAgent(config={"name": "Forensic_Unit_01"})

    cases = [
        {
            "company": "Enron 2.0 (Energy AI)",
            "sector": "Energy / Tech",
            "data": {
                "revenue": 100000000,
                "cash_flow": 5000000,  # 5% of revenue (Suspicious)
                "expenses": 80000000,
                "net_income": 20000000,
                "growth_rate": 0.60,    # 60% growth (Suspicious)
                "liabilities": 50000000
            },
            "description": "Rapidly growing energy trading firm claiming AI-driven arbitrage profits."
        },
        {
            "company": "Wirecard AI Payments",
            "sector": "Fintech",
            "data": {
                "revenue": 50000000,
                "cash_flow": 40000000,
                "expenses": 10000000,
                "net_income": 40000000,
                "growth_rate": 0.20,
                "cash_reserves_claimed": 2000000000, # 2B
                "cash_reserves_audited": 10000       # Missing
            },
            "description": "Payment processor claiming massive Asian subsidiary revenues."
        },
        {
            "company": "FTX Reborn",
            "sector": "Crypto / Exchange",
            "data": {
                "revenue": 0,           # Pre-revenue?
                "expenses": 0,          # No expenses reported?
                "net_income": 0,
                "token_valuation": 5000000000, # 5B valuation based on nothing
                "customer_deposits": 1000000000,
                "liquid_assets": 5000000
            },
            "description": "Next-gen exchange with 'fully on-chain' auditing (that no one can find)."
        }
    ]

    results = []

    print("Running Forensic Audit...")
    for case in cases:
        print(f"Analyzing {case['company']}...")

        # 1. Detect Anomalies
        # The agent's detect_anomalies method expects a dict.
        # It modifies self.anomalies_detected, but also returns them.
        anomalies = agent.detect_anomalies(case['data'])

        # 2. Restate Financials
        restated = agent.restate_financials(case['data'])

        # Calculate impact
        original_rev = float(case['data'].get('revenue', 0))
        restated_rev = float(restated.get('revenue', 0))
        impact_pct = ((restated_rev - original_rev) / original_rev * 100) if original_rev else -100.0

        results.append({
            "company": case['company'],
            "sector": case['sector'],
            "description": case['description'],
            "original_financials": case['data'],
            "restated_financials": restated,
            "anomalies": anomalies,
            "impact_assessment": {
                "revenue_impact_pct": round(impact_pct, 2),
                "risk_score": len(anomalies) * 25, # Simple scoring
                "verdict": "FRAUD_LIKELY" if anomalies else "CLEAN"
            }
        })

    output_path = "showcase/data/fraud_cases.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} fraud cases to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
