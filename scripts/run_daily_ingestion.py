import asyncio
import logging
import os
import sys

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agents.sub_agents.data_ingestion_agent import DataIngestionAgent

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Daily Ingestion Script (Adam v24.0 Sprint 1)...")

    # Configuration
    config = {
        "agent_id": "data_ingestion_01",
        "storage_path": "data"
    }

    # Instantiate Agent
    agent = DataIngestionAgent(config)

    # Test Tickers (Subset of S&P 500 for demo)
    # Using 'AAPL', 'MSFT', 'NVDA' as requested by typical users for testing
    tickers = ["AAPL", "MSFT", "NVDA"]
    logger.info(f"Target Tickers: {tickers}")

    # 1. Daily Ingestion (Historical)
    logger.info("--- Task 1: Ingest Daily History ---")
    result = await agent.execute("ingest_daily", tickers=tickers, period="1mo") # Short period for speed in test
    logger.info(f"Result: {result}")

    # Verify Daily Files
    # StorageEngine stores daily at: data/daily/region=US/year=2024/...
    # But since we don't know the exact year partitions without listing, we check the root.
    daily_path = os.path.join("data", "daily", "region=US")
    if os.path.exists(daily_path):
        logger.info(f"Verified: Daily data directory exists at {daily_path}")
        # List subdirs to see years
        try:
            years = os.listdir(daily_path)
            logger.info(f"Years found: {years}")
        except Exception as e:
            logger.warning(f"Could not list years: {e}")
    else:
        logger.error(f"Failed Verification: Daily data directory not found at {daily_path}")

    # 2. Intraday Ingestion (Eager 1m)
    logger.info("--- Task 2: Ingest Intraday (1m) ---")
    result = await agent.execute("ingest_intraday", tickers=tickers)
    logger.info(f"Result: {result}")

    # Verify Intraday Files
    # StorageEngine stores at: data/intraday/frequency=1m/ticker=AAPL/...
    for ticker in tickers:
        intraday_path = os.path.join("data", "intraday", "frequency=1m", f"ticker={ticker}")
        if os.path.exists(intraday_path):
             logger.info(f"Verified: Intraday data for {ticker} exists at {intraday_path}")
        else:
             logger.error(f"Failed Verification: Intraday data for {ticker} not found at {intraday_path}")

    logger.info("Daily Ingestion Script Completed.")

if __name__ == "__main__":
    asyncio.run(main())
