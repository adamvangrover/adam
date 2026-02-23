"""
Example 04: Environment Rotation
--------------------------------
Demonstrates switching between different execution engines (Simulation vs. Live)
via configuration, proving portability and modularity.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine.live_mock_engine import LiveMockEngine
# Using a mock class for the "Real" engine to demonstrate the pattern
class RealTradingEngine:
    def get_market_pulse(self):
        return {"indices": {"SPX": {"price": 4200.00, "source": "REAL_FEED"}}}

class EngineFactory:
    @staticmethod
    def get_engine(environment: str):
        if environment == "LIVE":
            print(">>> [Factory] initializing REAL TRADING ENGINE")
            return RealTradingEngine()
        elif environment == "SIMULATION":
            print(">>> [Factory] initializing SIMULATION ENGINE")
            return LiveMockEngine()
        else:
            raise ValueError(f"Unknown environment: {environment}")

def run_rotation_demo():
    # 1. Run in Simulation
    print("\n--- ROTATION 1: SIMULATION ---")
    env_sim = "SIMULATION"
    engine_sim = EngineFactory.get_engine(env_sim)
    data_sim = engine_sim.get_market_pulse()
    print(f"Data: {data_sim['indices']['SPX']}")

    # 2. Run in Live (Mocked Real)
    print("\n--- ROTATION 2: LIVE PRODUCTION ---")
    env_live = "LIVE"
    engine_live = EngineFactory.get_engine(env_live)
    data_live = engine_live.get_market_pulse()
    print(f"Data: {data_live['indices']['SPX']}")

if __name__ == "__main__":
    run_rotation_demo()
