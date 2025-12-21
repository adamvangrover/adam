from typing import Any, Dict, List, Optional
import logging
from core.agents.agent_base import AgentBase
from core.gold_standard.ingestion import IngestionEngine
from core.gold_standard.storage import StorageEngine

logger = logging.getLogger(__name__)


class DataIngestionAgent(AgentBase):
    """
    Agent responsible for data ingestion tasks using the Gold Standard Toolkit.
    Handles daily history downloads, intraday snapshots, and schema validation.

    Version: Adam v24 (Sprint 1: Sensory Layer)
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Any = None):
        super().__init__(config, constitution, kernel)

        # Initialize the Gold Standard Storage and Ingestion engines
        # Default to 'data' directory if not specified in config
        storage_path = self.config.get("storage_path", "data")
        self.storage = StorageEngine(base_path=storage_path)

        # Using IngestionEngine (Sprint 1 Standard)
        self.ingestion_engine = IngestionEngine(self.storage)
        logger.info(f"DataIngestionAgent initialized with storage path: {storage_path}")

    async def execute(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes data ingestion tasks.

        Supported tasks:
        - ingest_daily: Batch download of daily history.
        - ingest_intraday: Eager ingestion of 1m data (last 7 days).
        - get_snapshot: Get realtime snapshot for a single ticker.
        """
        try:
            logger.info(f"DataIngestionAgent executing task: {task}")

            if task == "ingest_daily":
                tickers = kwargs.get("tickers", [])
                if not tickers:
                    return {"status": "error", "message": "No tickers provided"}

                # Feature branch supports configurable period/interval
                period = kwargs.get("period", "max")
                interval = kwargs.get("interval", "1d")

                logger.info(f"Starting daily ingestion for {len(tickers)} tickers (Period: {period})...")
                self.ingestion_engine.ingest_daily_history(tickers, period=period, interval=interval)
                return {"status": "success", "message": f"Completed daily ingestion for {len(tickers)} tickers"}

            elif task == "ingest_intraday":
                tickers = kwargs.get("tickers", [])
                if not tickers:
                    return {"status": "error", "message": "No tickers provided"}

                logger.info(f"Starting intraday ingestion for {len(tickers)} tickers...")
                self.ingestion_engine.ingest_intraday_eager(tickers)
                return {"status": "success", "message": f"Completed intraday ingestion for {len(tickers)} tickers"}

            elif task == "get_snapshot":
                # Feature branch schema specifies single ticker for snapshot
                ticker = kwargs.get("ticker")
                if not ticker:
                    return {"status": "error", "message": "No ticker provided"}

                snapshot = self.ingestion_engine.get_realtime_snapshot(ticker)
                return {"status": "success", "data": snapshot}

            else:
                return {"status": "error", "message": f"Unknown task: {task}"}

        except Exception as e:
            logger.error(f"Error executing task {task}: {e}")
            return {"status": "error", "message": str(e)}

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the skills exposed to the Kernel/LLM.
        """
        return {
            "name": "DataIngestionAgent",
            "description": "Handles ingestion of financial data (daily history, intraday, snapshots).",
            "skills": [
                {
                    "name": "ingest_daily",
                    "description": "Ingest daily historical data for a list of tickers.",
                    "parameters": {
                        "tickers": "List of ticker symbols",
                        "period": "Data period (default: max)",
                        "interval": "Data interval (default: 1d)"
                    }
                },
                {
                    "name": "ingest_intraday",
                    "description": "Ingest 1-minute intraday data for the last 7 days.",
                    "parameters": {
                        "tickers": "List of ticker symbols"
                    }
                },
                {
                    "name": "get_snapshot",
                    "description": "Get real-time market snapshot for a ticker.",
                    "parameters": {
                        "ticker": "Ticker symbol"
                    }
                }
            ]
        }
