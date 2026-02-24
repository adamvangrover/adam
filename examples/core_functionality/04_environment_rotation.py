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

from core.engine.factory import EngineFactory

def run_rotation_demo():
    print(">>> Environment Rotation Demo")
    print(">>> Adam v26.0 supports hot-swapping execution backends.")
    print("-" * 50)

    # 1. Run in Simulation (System 3)
    # Uses LiveMockEngine with generated data
    print("\n--- ROTATION 1: SIMULATION ENGINE (System 3) ---")
    env_sim = "SIMULATION"
    engine_sim = EngineFactory.get_engine(env_sim)

    pulse_sim = engine_sim.get_market_pulse()
    print(f"[{engine_sim.__class__.__name__}] Market Pulse: {pulse_sim['indices']['SPX']['price']} (Simulated)")

    memo_sim = engine_sim.generate_credit_memo("TSLA", "Tesla Inc", "Auto")
    print(f"[{engine_sim.__class__.__name__}] Memo Mode: {memo_sim['mode']}")


    # 2. Run in Live Production (System 1/2)
    # Uses RealTradingEngine (Stub) connecting to live feeds
    print("\n--- ROTATION 2: LIVE PRODUCTION ENGINE (System 1) ---")
    env_live = "LIVE"
    engine_live = EngineFactory.get_engine(env_live)

    pulse_live = engine_live.get_market_pulse()
    print(f"[{engine_live.__class__.__name__}] Market Pulse: {pulse_live['indices']['SPX']['price']} (Live Feed Stub)")

    memo_live = engine_live.generate_credit_memo("AAPL", "Apple Inc", "Tech")
    print(f"[{engine_live.__class__.__name__}] Memo Mode: {memo_live['mode']}")
    print(f"[{engine_live.__class__.__name__}] Note: {memo_live['memo']}")

if __name__ == "__main__":
    run_rotation_demo()
