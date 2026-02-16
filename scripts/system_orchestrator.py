#!/usr/bin/env python3
"""
ADAM v24.0 :: SYSTEM ORCHESTRATOR
-----------------------------------------------------------------------------
The "Master Controller" for daily system updates, autonomous cycles, and
integrity verification. Ensures robustness, stability, and continuous evolution.
-----------------------------------------------------------------------------
"""

import os
import sys
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Setup Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "system_orchestrator.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SystemOrchestrator")

# Constants
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

class SystemOrchestrator:
    def __init__(self):
        self.start_time = datetime.now()
        logger.info(f"Initialized System Orchestrator at {self.start_time}")

    def run_script(self, script_name, args=None, description="Task"):
        """Executes a python script and logs output."""
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        logger.info(f"STARTING: {description} ({script_name})")
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT
            )

            if result.returncode == 0:
                logger.info(f"SUCCESS: {description}")
                if result.stdout:
                    logger.debug(f"STDOUT:\n{result.stdout.strip()}")
                return True
            else:
                logger.error(f"FAILURE: {description}")
                logger.error(f"STDERR:\n{result.stderr.strip()}")
                return False

        except Exception as e:
            logger.exception(f"EXCEPTION during {description}: {e}")
            return False

    def run_daily_cycle(self):
        """Orchestrates the full daily update cycle."""
        logger.info(">>> BEGINNING DAILY UPDATE CYCLE <<<")
        success_count = 0
        total_tasks = 5

        # 1. Data Ingestion
        if self.run_script("run_daily_ingestion.py", description="Daily Data Ingestion"):
            success_count += 1

        # 2. Market Snapshot Generation
        if self.run_script("generate_market_snapshot.py", description="Market Snapshot Generation"):
            success_count += 1

        # 3. Market Mayhem Report Generation
        if self.run_script("generate_market_mayhem_archive.py", description="Market Mayhem Reports"):
            success_count += 1

        # 4. System Brain Data Update
        if self.run_script("generate_system_brain_data.py", description="System Brain Update"):
            success_count += 1

        # 5. Module Export Update
        if self.run_script("export_module.py", args=["all", "--output", "exports"], description="Module Exports"):
            success_count += 1

        # Summary
        duration = datetime.now() - self.start_time
        logger.info(">>> DAILY CYCLE COMPLETE <<<")
        logger.info(f"Duration: {duration}")
        logger.info(f"Success Rate: {success_count}/{total_tasks}")

        if success_count == total_tasks:
            return True
        else:
            return False

if __name__ == "__main__":
    orchestrator = SystemOrchestrator()
    orchestrator.run_daily_cycle()
