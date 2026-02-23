"""
Example 02: Compute Layer Standalone
------------------------------------
Demonstrates running the simulation/compute engine (Compute Layer) in isolation.
"""

import sys
import os
import json

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine.live_mock_engine import LiveMockEngine
from core.system.provenance_logger import ProvenanceLogger, ActivityType

def run_compute_cycle():
    logger = ProvenanceLogger()
    engine = LiveMockEngine()

    print(">>> Initializing Compute Layer (LiveMockEngine)...")

    # 1. Get Market Pulse (Simulation Step)
    pulse = engine.get_market_pulse()
    print(f"SPX Price: {pulse['indices']['SPX']['price']}")

    logger.log_activity(
        agent_id="LiveMockEngine",
        activity_type=ActivityType.SIMULATION,
        input_data={"command": "get_market_pulse"},
        output_data=pulse,
        data_source="SimulatedMarketData"
    )

    # 2. Generate Credit Memo (Compute Intensive)
    print(">>> Generating Synthetic Credit Memo...")
    ticker = "TSLA"
    memo = engine.generate_credit_memo(ticker, name="Tesla Inc", sector="Auto")

    print(f"Memo Leverage: {memo['data']['financials']['leverage']}x")
    print(f"Memo Recommendation: {memo['memo'].split('System Conclusion')[1].strip()}")

    logger.log_activity(
        agent_id="CreditComputer",
        activity_type=ActivityType.CALCULATION,
        input_data={"ticker": ticker, "model": "ICAT-Lite"},
        output_data=memo['data'],
        data_source="InternalCompute"
    )

if __name__ == "__main__":
    run_compute_cycle()
    print(">>> Compute cycle complete. Logs generated.")
