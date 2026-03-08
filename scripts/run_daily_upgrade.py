#!/usr/bin/env python3
"""
ADAM v24.0 :: DAILY UPGRADE SWARM SCHEDULER
-----------------------------------------------------------------------------
Executes the systematic 7-phase daily upgrade loop by invoking the UpgradeManager.
Can be run manually or triggered via cron jobs / CI workflows.
-----------------------------------------------------------------------------
"""
import asyncio
import logging
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine.upgrade_manager import UpgradeManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DailyUpgradeRunner")

async def main():
    logger.info(">>> BEGINNING SYSTEMATIC UPGRADE SWARM CYCLE <<<")
    manager = UpgradeManager(tracker_path="data/upgrade_targets.json")
    await manager.execute_daily_cycle()
    logger.info(">>> CYCLE COMPLETE <<<")

if __name__ == "__main__":
    asyncio.run(main())
