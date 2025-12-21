import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.advisory.robo_advisor import RoboAdvisor
from core.trading.hft.hft_engine import HFTStrategy


def test_hft_init():
    print("[TEST] Initializing HFT Engine...")
    try:
        engine = HFTStrategy(symbol="TEST", balance=10000.0)
        assert engine.cash == 10000.0
        print("PASS: HFT Engine initialized.")
    except Exception as e:
        print(f"FAIL: HFT Engine init failed: {e}")
        sys.exit(1)

def test_robo_advisor():
    print("[TEST] Running Robo-Advisor Logic...")
    try:
        advisor = RoboAdvisor("data/strategies/gold_standard_portfolio.json")

        # Test Case 1: High Risk
        inputs = {'time_horizon': 30, 'liquidity_needs': 'Low', 'market_drop_reaction': 100}
        rec = advisor.generate_recommendation(inputs)
        assert rec['recommendation']['band'] == "AGGRESSIVE"
        print("PASS: Robo-Advisor logic (Aggressive).")

        # Test Case 2: Low Capacity Override
        inputs = {'time_horizon': 2, 'liquidity_needs': 'High', 'market_drop_reaction': 100}
        rec = advisor.generate_recommendation(inputs)
        assert rec['recommendation']['band'] == "CONSERVATIVE"
        assert rec['recommendation']['warning'] is not None
        print("PASS: Robo-Advisor logic (Suitability Override).")

    except Exception as e:
        print(f"FAIL: Robo-Advisor failed: {e}")
        sys.exit(1)

def test_portfolio_json():
    print("[TEST] Verifying Gold Standard JSON...")
    path = "data/strategies/gold_standard_portfolio.json"
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            assert "equities" in data['target_allocation']
            assert "fixed_income" in data['target_allocation']
            assert "alternatives" in data['target_allocation']
        print("PASS: Portfolio JSON is valid.")
    except Exception as e:
        print(f"FAIL: JSON verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_hft_init()
    test_portfolio_json()
    test_robo_advisor()
    print("\nALL MODULES VERIFIED.")
