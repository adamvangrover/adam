"""
Example 02: Compute Layer Standalone
------------------------------------
Demonstrates running the simulation/compute engine (Compute Layer) in isolation.
Uses the EngineFactory to provision a computational backend (System 3).
"""

import sys
import os
import json

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine.factory import EngineFactory
from core.system.provenance_logger import ProvenanceLogger, ActivityType

def run_compute_cycle():
    logger = ProvenanceLogger()

    # 1. Provision Compute Engine (Simulation Mode)
    # The compute layer is responsible for world modeling and simulation.
    print(">>> Initializing Compute Layer (EngineFactory: SIMULATION)...")
    engine = EngineFactory.get_engine("SIMULATION")

    # 2. Execute Simulation Step (Market Pulse)
    pulse = engine.get_market_pulse()
    spx_price = pulse['indices']['SPX']['price']
    print(f"[{engine.__class__.__name__}] Market Pulse: SPX @ {spx_price}")

    logger.log_activity(
        agent_id="ComputeLayer",
        activity_type=ActivityType.SIMULATION,
        input_data={"command": "get_market_pulse"},
        output_data={"spx_price": spx_price, "timestamp": pulse['timestamp']},
        data_source="SimulatedMarketData"
    )

    # 3. Execute Deep Compute (Credit Memo Generation)
    # This simulates a heavy computational task (ICAT Engine)
    print(">>> Generating Synthetic Credit Memo (Compute Job)...")
    ticker = "AMZN"
    memo = engine.generate_credit_memo(ticker, name="Amazon.com Inc", sector="Consumer Discretionary")

    leverage = memo['data']['financials']['leverage']
    print(f"[{engine.__class__.__name__}] Compute Result: Leverage {leverage:.2f}x")
    print(f"[{engine.__class__.__name__}] Memo Conclusion: {memo['memo'].split('System Conclusion')[1].strip()}")

    logger.log_activity(
        agent_id="ComputeLayer",
        activity_type=ActivityType.CALCULATION,
        input_data={"ticker": ticker, "task": "credit_memo_generation"},
        output_data=memo['data'],
        data_source="InternalCompute",
        capture_full_io=True
    )

if __name__ == "__main__":
    run_compute_cycle()
    print(">>> Compute cycle complete. Provenance logs generated.")
