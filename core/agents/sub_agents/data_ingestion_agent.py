from typing import Any, Dict, List, Optional
import logging
from core.agents.agent_base import AgentBase
from core.gold_standard.data_fetcher import DataFetcher
from core.gold_standard.storage import StorageEngine

logger = logging.getLogger(__name__)

class DataIngestionAgent(AgentBase):
    """
    DataIngestionAgent (Sprint 1: Sensory Layer).
    Responsible for executing daily and eager data ingestion tasks using the DataFetcher.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # Initialize DataFetcher with default storage location
        self.storage = StorageEngine()
        self.fetcher = DataFetcher(self.storage)

    async def execute(self, task: str = "daily", tickers: List[str] = [], **kwargs) -> Dict[str, Any]:
        """
        Executes ingestion tasks.
        
        Args:
            task (str): 'daily' for batch history, 'eager' for 1m intraday, 'snapshot' for real-time.
            tickers (List[str]): List of ticker symbols to process.
        """
        logger.info(f"DataIngestionAgent received task: {task} for {len(tickers)} tickers.")
        
        if not tickers:
            return {"status": "error", "message": "No tickers provided."}

        try:
            if task == "daily":
                self.fetcher.ingest_daily_history(tickers)
                return {"status": "success", "message": f"Daily ingestion completed for {len(tickers)} tickers."}
            
            elif task == "eager":
                self.fetcher.ingest_intraday_eager(tickers)
                return {"status": "success", "message": f"Eager intraday ingestion completed for {len(tickers)} tickers."}
            
            elif task == "snapshot":
                results = {}
                for ticker in tickers:
                    results[ticker] = self.fetcher.get_realtime_snapshot(ticker)
                return {"status": "success", "data": results}
            
            else:
                return {"status": "error", "message": f"Unknown task: {task}"}
                
        except Exception as e:
            logger.error(f"Error executing ingestion task: {e}")
            return {"status": "error", "message": str(e)}
