#!/usr/bin/env python3
import asyncio
import logging
import sys
import os

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.sub_agents.data_ingestion_agent import DataIngestionAgent

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    # Configuration
    config = {
        "name": "DataIngestionAgent",
        "agent_id": "ingest_001",
        "description": "Daily S&P 500 Ingestion"
    }
    
    # Initialize Agent
    agent = DataIngestionAgent(config)
    
    # Target Tickers (Subset of S&P 500 for Sprint 1 verification)
    # Ideally, this would be fetched from 'data/sp500_tickers.json' or similar
    tickers = ["SPY", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM"]
    
    logging.info(f"Starting Daily Ingestion for {len(tickers)} tickers...")
    
    # 1. Run Daily History Ingestion (Batch)
    logging.info("--- Phase 1: Daily History ---")
    result_daily = await agent.execute(task="daily", tickers=tickers)
    logging.info(f"Daily Result: {result_daily}")
    
    # 2. Run Eager Intraday Ingestion (1m)
    logging.info("--- Phase 2: Eager Intraday (1m) ---")
    result_eager = await agent.execute(task="eager", tickers=tickers)
    logging.info(f"Eager Result: {result_eager}")
    
    logging.info("Ingestion Cycle Complete.")

if __name__ == "__main__":
    asyncio.run(main())
