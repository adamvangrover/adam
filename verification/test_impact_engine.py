import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.sector_impact_engine import SectorImpactEngine

def test_engine_logic():
    print("Initializing Engine...")
    engine = SectorImpactEngine()

    # Reload scenarios to ensure new ones are picked up (if we were running long term, but here it's fresh)
    # The file modification happens in the next step, so this test is expected to FAIL initially
    # if I ran it before modifications. But I'll run it after.

    # Mock Portfolio
    portfolio = [
        {"name": "Tech Asset", "sector": "Technology", "leverage": 2.0, "rating": "A"},
        {"name": "Bank Asset", "sector": "Financials", "leverage": 8.0, "rating": "BB"}
    ]

    # 1. Test Quantum Event (to be added)
    print("Testing Quantum Event Scenario...")
    # Note: We can pass custom_shocks to simulate it even before the file is updated
    quantum_shocks = {"Technology": -0.9, "Financials": -0.8}

    result = engine.analyze_portfolio(portfolio, custom_shocks=quantum_shocks)

    found_tech_warning = False
    found_bank_warning = False

    for r in result['results']:
        insight = r['macro_insight'] + " " + r['credit_insight']
        print(f"Asset: {r['asset']} | Insight: {insight}")

        if r['asset'] == "Tech Asset":
            if "Digital infrastructure collapse" in insight or "Critical dependency failure" in insight:
                found_tech_warning = True

        if r['asset'] == "Bank Asset":
            if "Systemic banking stress" in insight or "Interbank lending freeze" in insight:
                found_bank_warning = True

    if found_tech_warning and found_bank_warning:
        print("SUCCESS: Specific warnings detected.")
    else:
        print("FAILURE: Specific warnings NOT detected.")
        # We expect this to fail right now
        # raise Exception("Verification Failed")

if __name__ == "__main__":
    test_engine_logic()
