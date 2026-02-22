import asyncio
import logging
from core.agents.risk_assessment_agent import RiskAssessmentAgent

# Setup logging
logging.basicConfig(level=logging.INFO)

async def run_verification():
    print("Initializing RiskAssessmentAgent...")
    config = {"persona": "Risk Analyst", "description": "Test Agent"}
    agent = RiskAssessmentAgent(config)

    # Mock Target Data
    target_data = {
        "company_name": "TEST_RISK_CORP",
        "financial_data": {
            "industry": "Technology",
            "credit_rating": "BBB", # Should map to ~0.5% default prob
            # Liquidity Data
            "liquidity_ratio": 0.8, # Low
            "cash": 500,
            "monthly_burn_rate": 50, # 10 months runway
            # Recovery Data
            "total_assets": 2000,
            "total_debt": 1000 # 2.0x coverage -> high recovery
        },
        "market_data": {
            "price_data": [100, 101, 99, 102, 98], # Low vol
            "trading_volume": 1000000
        }
    }

    print("Executing Risk Assessment (Investment/Credit)...")
    # We expect the agent to handle 'investment' type but with new credit focus
    result = await agent.execute(target_data, risk_type="investment")

    print(f"Result Keys: {list(result.keys())}")
    factors = result.get("risk_factors", {})
    print(f"Risk Factors: {list(factors.keys())}")

    # Assertions for new mandates
    assert "default_probability" in factors, "Default Probability missing"
    assert "recovery_rate" in factors, "Recovery Rate missing"
    assert "liquidity_runway" in factors, "Liquidity Runway missing"

    # Check values
    print(f"Default Prob: {factors['default_probability']}")
    print(f"Recovery Rate: {factors['recovery_rate']}")
    print(f"Liquidity Runway: {factors['liquidity_runway']}")

    # Basic logic checks
    # Recovery rate should be high (Assets > Debt)
    # Liquidity runway should be ~10 months

    if factors['recovery_rate'] > 0.8:
        print("SUCCESS: Recovery rate logic seems correct (high coverage).")
    else:
        print(f"WARNING: Recovery rate {factors['recovery_rate']} seems low for 2.0x coverage.")

    if 9 <= factors['liquidity_runway'] <= 11:
         print("SUCCESS: Liquidity runway calculation seems correct (10 months).")
    else:
         print(f"WARNING: Liquidity runway {factors['liquidity_runway']} != 10.")

if __name__ == "__main__":
    asyncio.run(run_verification())
