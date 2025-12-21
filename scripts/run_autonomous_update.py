import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agents.sub_agents.data_ingestion_agent import DataIngestionAgent
from core.data_sources.data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutonomousUpdater")

TEMPLATE_PATH = "data/user_universe.json"

class AutonomousUpdater:
    def __init__(self, template_path):
        self.template_path = template_path
        self.config = self._load_template()
        self.agent = DataIngestionAgent({
            "name": "AutonomousIngestor",
            "storage_path": "data"
        })
        self.fetcher = DataFetcher()

    def _load_template(self):
        if not os.path.exists(self.template_path):
            logger.error(f"Template not found at {self.template_path}")
            return {}
        with open(self.template_path, 'r') as f:
            return json.load(f)

    def _save_template(self):
        self.config["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.template_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info("Universe template updated.")

    def get_all_tickers(self):
        universe = self.config.get("universe", {})
        all_tickers = []
        for category, tickers in universe.items():
            all_tickers.extend(tickers)
        return list(set(all_tickers))

    async def run_ingestion(self):
        tickers = self.get_all_tickers()
        defaults = self.config.get("configurations", {})

        logger.info(f"Loaded {len(tickers)} tickers from template.")

        # 1. Ingest Daily History
        logger.info("--- Phase 1: Historical Data Ingestion ---")
        await self.agent.execute(
            "ingest_daily",
            tickers=tickers,
            period=defaults.get("default_period", "1mo"),
            interval=defaults.get("default_interval", "1d")
        )

        # 2. Ingest Real-time Snapshots (Intraday)
        logger.info("--- Phase 2: Intraday Snapshot Ingestion ---")
        await self.agent.execute("ingest_intraday", tickers=tickers)

    async def run_discovery(self):
        """Autonomous step to find and add new tickers based on criteria."""
        if not self.config.get("configurations", {}).get("enable_discovery", False):
            return

        logger.info("--- Phase 3: Autonomous Discovery ---")

        # Example Logic: Fetch a sector ETF's top holdings or similar
        # For this example, we mock a 'trending' discovery
        # In a real scenario, you could use self.fetcher to scan volume gainers

        # Mock: Let's say we discovered 'COIN' is trending
        discovered_ticker = "COIN"

        current_watch = self.config["universe"].get("watch_list", [])
        if discovered_ticker not in current_watch:
            logger.info(f"Discovered new trending asset: {discovered_ticker}")
            current_watch.append(discovered_ticker)
            self.config["universe"]["watch_list"] = current_watch
            self._save_template()

            # Immediately ingest the new find
            await self.agent.execute("ingest_daily", tickers=[discovered_ticker], period="1mo")

async def main():
    updater = AutonomousUpdater(TEMPLATE_PATH)
    await updater.run_ingestion()
    await updater.run_discovery()

if __name__ == "__main__":
    asyncio.run(main())
