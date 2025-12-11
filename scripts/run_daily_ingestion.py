#!/usr/bin/env python3
import asyncio
import logging
import os
import sys

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agents.sub_agents.data_ingestion_agent import DataIngestionAgent

# Configure Logging (Adam v24 Standard)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Daily Ingestion Script (Adam v24.0 Sprint 1)...")

    # Configuration
    # Merging metadata from main with functional config from feature branch
    config = {
        "name": "DataIngestionAgent",
        "agent_id": "data_ingestion_01",
        "description": "Daily S&P 500 Ingestion",
        "storage_path": "data"
    }

    # Instantiate Agent
    agent = DataIngestionAgent(config)

    # Target Tickers
    # Using the broader list from 'main' for better coverage, but limiting period for speed
    tickers = ["SPY", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM"]
    logger.info(f"Target Tickers: {tickers}")

    # 1. Daily Ingestion (Historical)
    logger.info("--- Task 1: Ingest Daily History ---")
    # Using 'ingest_daily' (Feature branch convention) and '1mo' period
    result = await agent.execute("ingest_daily", tickers=tickers, period="1mo")
    logger.info(f"Result: {result}")

    # Verify Daily Files
    # StorageEngine stores daily at: data/daily/region=US/year=2024/...
    daily_path = os.path.join("data", "daily", "region=US")
    if os.path.exists(daily_path):
        logger.info(f"Verified: Daily data directory exists at {daily_path}")
        try:
            years = os.listdir(daily_path)
            logger.info(f"Years found: {years}")
        except Exception as e:
            logger.warning(f"Could not list years: {e}")
    else:
        logger.error(f"Failed Verification: Daily data directory not found at {daily_path}")

    # 2. Intraday Ingestion (Eager 1m)
    logger.info("--- Task 2: Ingest Intraday (1m) ---")
    # Using 'ingest_intraday' (Feature branch convention)
    result = await agent.execute("ingest_intraday", tickers=tickers)
    logger.info(f"Result: {result}")

    # Verify Intraday Files
    # StorageEngine stores at: data/intraday/frequency=1m/ticker=AAPL/...
    # Verifying a subset to avoid log spam
    for ticker in tickers[:3]: 
        intraday_path = os.path.join("data", "intraday", "frequency=1m", f"ticker={ticker}")
        if os.path.exists(intraday_path):
             logger.info(f"Verified: Intraday data for {ticker} exists at {intraday_path}")
        else:
             logger.error(f"Failed Verification: Intraday data for {ticker} not found at {intraday_path}")

    logger.info("Daily Ingestion Script Completed.")

if __name__ == "__main__":
    asyncio.run(main())