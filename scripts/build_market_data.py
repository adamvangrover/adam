import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.utils.logging_utils import setup_logging, get_logger
from core.financial_data import MarketDiscoveryAgent, DataLakehouse

logger = get_logger("market_data_builder")

def main():
    parser = argparse.ArgumentParser(description="Adam Financial Framework: Market Data Builder")
    parser.add_argument("--query", type=str, default="Technology", help="Search query for discovery (e.g., 'ESG', 'Semiconductors')")
    parser.add_argument("--limit", type=int, default=10, help="Max number of tickers to discover")
    parser.add_argument("--period", type=str, default="1y", help="Historical data period (e.g., '1y', 'max')")
    parser.add_argument("--lake-path", type=str, default="data/market_lakehouse", help="Path to Data Lakehouse")

    args = parser.parse_args()

    setup_logging()

    logger.info("Starting Market Data Build Process...")

    # 1. Discovery
    discovery_agent = MarketDiscoveryAgent()
    tickers = discovery_agent.search_universe(args.query, limit=args.limit)

    if not tickers:
        logger.warning("No tickers found. Exiting.")
        return

    logger.info(f"Discovered {len(tickers)} tickers for query '{args.query}': {[t.symbol for t in tickers]}")

    # 2. Storage Initialization
    lakehouse = DataLakehouse(root_path=args.lake_path)

    # 3. Store Metadata
    logger.info("Storing metadata...")
    lakehouse.store_metadata(tickers)

    # 4. Ingest Price Data
    logger.info("Ingesting historical price data...")
    lakehouse.ingest_tickers(tickers, period=args.period)

    logger.info("Market Data Build Complete.")
    logger.info(f"Data stored in {args.lake_path}")

if __name__ == "__main__":
    main()
