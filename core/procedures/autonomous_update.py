# core/procedures/autonomous_update.py

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

# Leveraging existing robust components
from core.agents.sub_agents.data_ingestion_agent import DataIngestionAgent

# Assuming DataFetcher path based on context, though mostly managed by the agent now

logger = logging.getLogger("adam.core.procedures.autonomous_update")

class RoutineMaintenance:
    """
    Encapsulates standard maintenance procedures for the Adam Financial OS.
    Designed to be stateless where possible, loaded by the TemporalEngine.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.universe_path = os.path.join(data_dir, "user_universe.json")

        # Initialize the specialist agent for heavy lifting
        self.ingestion_agent = DataIngestionAgent({
            "name": "RoutineIngestor",
            "storage_path": data_dir
        })

    async def run_market_data_refresh(self, period: str = "1d", interval: str = "1m"):
        """
        Performs a 'Tick' update: getting the latest intraday data for the universe.
        """
        logger.info("Initiating Market Data Refresh Routine...")

        tickers = self._get_universe_tickers()
        if not tickers:
            logger.warning("No tickers found in universe. Aborting refresh.")
            return

        # Execute ingestion via the Agent (reusing the logic from the script)
        # We assume the agent has an 'ingest_intraday' capability or we map to 'ingest_daily'
        # based on the script provided.

        try:
            # Using the exact command structure from the original script
            await self.ingestion_agent.execute("ingest_intraday", tickers=tickers)
            logger.info("Market Data Refresh Routine Complete.")
        except Exception as e:
            logger.error(f"Market Data Refresh failed: {e}")

    async def run_deep_discovery(self):
        """
        Performs a 'Deep' update: looking for new assets and updating long-term history.
        """
        logger.info("Initiating Deep Discovery Routine...")

        # Logic adapted from run_autonomous_update.py
        # Here we would expand to actually check "trending" lists via an API
        # For now, we maintain the mock logic but documented as a placeholder for expansion

        discovered_ticker = "COIN" # Placeholder from original script

        universe_data = self._load_universe()
        watch_list = universe_data.get("universe", {}).get("watch_list", [])

        if discovered_ticker not in watch_list:
            logger.info(f"Discovery Agent found new asset: {discovered_ticker}")
            watch_list.append(discovered_ticker)

            # Update the file safely
            if "universe" not in universe_data: universe_data["universe"] = {}
            universe_data["universe"]["watch_list"] = watch_list
            universe_data["metadata"]["last_updated"] = datetime.now().isoformat()

            self._save_universe(universe_data)

            # Immediate ingestion for the new asset
            await self.ingestion_agent.execute("ingest_daily", tickers=[discovered_ticker], period="1mo")

    def _load_universe(self) -> Dict:
        if not os.path.exists(self.universe_path):
            return {"metadata": {}, "universe": {"watch_list": []}}
        try:
            with open(self.universe_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load universe: {e}")
            return {}

    def _save_universe(self, data: Dict):
        try:
            with open(self.universe_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save universe: {e}")

    def _get_universe_tickers(self) -> List[str]:
        data = self._load_universe()
        universe = data.get("universe", {})
        all_tickers = []
        for category, tickers in universe.items():
            if isinstance(tickers, list):
                all_tickers.extend(tickers)
        return list(set(all_tickers))
