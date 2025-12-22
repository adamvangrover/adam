# scripts/launch_system_pulse.py

import asyncio
import logging
import os
import sys

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.system.temporal_engine import TemporalEngine
from core.procedures.autonomous_update import RoutineMaintenance
from core.system.agent_orchestrator import AgentOrchestrator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("logs/adam_pulse.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdamPulse")

async def main():
    logger.info("Initializing Adam v23 System Pulse...")

    # 1. Initialize Core Systems
    # Orchestrator is loaded to ensure Agents are prepped and caches are hot
    orchestrator = AgentOrchestrator()

    # 2. Initialize the Temporal Engine (Async Scheduler)
    temporal_engine = TemporalEngine()

    # 3. Initialize Procedures
    maintenance = RoutineMaintenance()

    # 4. Register Tasks (The "Recurring Run")

    # Task A: Market Data Refresh (Fast Tick)
    # Runs every 15 minutes to capture intraday movements
    temporal_engine.register_task(
        name="Market Data Refresh",
        coro_func=maintenance.run_market_data_refresh,
        interval_seconds=900, # 15 minutes
        run_immediately=True  # Run once on startup
    )

    # Task B: Deep Discovery & Universe Expansion (Slow Tick)
    # Runs every 4 hours to check for new trending assets or structural updates
    temporal_engine.register_task(
        name="Deep Discovery",
        coro_func=maintenance.run_deep_discovery,
        interval_seconds=14400, # 4 hours
        run_immediately=False
    )

    # 5. Launch the Eternal Loop
    try:
        logger.info(">>> SYSTEM PULSE ACTIVE. PRESS CTRL+C TO HALT. <<<")
        await temporal_engine.start()
    except KeyboardInterrupt:
        logger.info("Manual stop signal received.")
    finally:
        temporal_engine.stop()
        logger.info("System Pulse shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
