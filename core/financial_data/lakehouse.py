import logging
import os
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import yfinance as yf
from .schema import MarketTicker

logger = logging.getLogger(__name__)

class DataLakehouse:
    """
    Manages the 'Gold Standard' static repository of market data.
    Uses Apache Parquet for efficient, columnar storage.
    """

    def __init__(self, root_path: str = "data/market_lakehouse"):
        self.root_path = Path(root_path)
        self.prices_path = self.root_path / "prices"
        self.metadata_path = self.root_path / "metadata"

        self._ensure_directories()

    def _ensure_directories(self):
        self.prices_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

    def ingest_tickers(self, tickers: List[Union[str, MarketTicker]], period: str = "10y", interval: str = "1d"):
        """
        Fetches historical data for a list of tickers and stores it in the Lakehouse.

        Args:
            tickers: List of ticker symbols or MarketTicker objects.
            period: Data period to download (e.g., "1y", "max").
            interval: Data interval (e.g., "1d", "1h").
        """
        symbols = [t.symbol if isinstance(t, MarketTicker) else t for t in tickers]

        if not symbols:
            logger.warning("No tickers provided for ingestion.")
            return

        logger.info(f"Ingesting data for {len(symbols)} tickers...")

        # Ingest sequentially to handle errors gracefully (Path A reliability)
        # Could be parallelized for Path B
        for symbol in symbols:
            try:
                self._ingest_single_ticker(symbol, period, interval)
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")

    def _ingest_single_ticker(self, symbol: str, period: str, interval: str):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return

        # Ensure index is standard timezone-naive or UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # Add symbol column for partitioning/filtering if needed later,
        # though we partition by directory currently.
        df['symbol'] = symbol

        # Save to Parquet with partitioning
        # Structure: root/prices/symbol=AAPL/data.parquet
        output_dir = self.prices_path / f"symbol={symbol}"
        output_dir.mkdir(exist_ok=True)

        file_path = output_dir / "data.parquet"

        # If exists, we might want to merge, but for now we overwrite/update based on blueprint
        # "edits only as needed... updating solely for new price action".
        # A simple approach is load existing, combine, deduplicate, save.

        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            combined = pd.concat([existing_df, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_parquet(file_path, compression='snappy')
            logger.info(f"Updated {symbol} data at {file_path}")
        else:
            df.to_parquet(file_path, compression='snappy')
            logger.info(f"Created {symbol} data at {file_path}")

    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Loads historical data for a symbol from the Lakehouse.
        """
        file_path = self.prices_path / f"symbol={symbol}" / "data.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for {symbol} in Lakehouse")

        return pd.read_parquet(file_path)

    def store_metadata(self, tickers: List[MarketTicker]):
        """
        Stores discovery metadata (sectors, industries) in a central Parquet file.
        """
        data = [t.model_dump(by_alias=True) for t in tickers]
        df = pd.DataFrame(data)

        if df.empty:
            return

        # Ensure correct types
        if 'discovery_date' in df.columns:
            df['discovery_date'] = pd.to_datetime(df['discovery_date'])

        output_path = self.metadata_path / "ticker_universe.parquet"

        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            combined = pd.concat([existing_df, df])
            # Deduplicate by symbol
            combined.drop_duplicates(subset=['symbol'], keep='last', inplace=True)
            combined.to_parquet(output_path, compression='snappy')
        else:
            df.to_parquet(output_path, compression='snappy')

        logger.info(f"Updated ticker universe metadata with {len(tickers)} entries")
