import logging
import uuid
from pathlib import Path
from typing import List, Union

import pandas as pd
import yfinance as yf

from .schema import MarketTicker

logger = logging.getLogger(__name__)

class DataLakehouse:
    """
    Manages the 'Gold Standard' static repository of market data.
    Uses Apache Parquet for efficient, columnar storage with Hive partitioning.
    """

    def __init__(self, root_path: str = "data"):
        self.root_path = Path(root_path)
        # Target: data/daily/region={region}/year={year}/{uuid}.parquet
        self.daily_path = self.root_path / "daily"
        self.metadata_path = self.root_path / "metadata"

        self._ensure_directories()

    def _ensure_directories(self):
        self.daily_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

    def _get_region(self, symbol: str) -> str:
        """Determines region from ticker symbol."""
        if "." not in symbol:
            return "US"
        suffix = symbol.split(".")[-1]
        # Common EU suffixes
        if suffix in ["L", "PA", "DE", "AM", "BR", "MI", "MC", "LS", "IR", "AS", "HE", "SW"]:
            return "EU"
        # Common APAC suffixes
        if suffix in ["HK", "T", "SS", "SZ", "KS", "AX", "NZ"]:
            return "APAC"
        return "ROW"

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

        df['symbol'] = symbol
        region = self._get_region(symbol)

        # Partition by Year
        df['year'] = df.index.year

        # We iterate by year to save partitioned files
        for year, group in df.groupby('year'):
            # Path: data/daily/region={region}/year={year}/{uuid}.parquet
            partition_dir = self.daily_path / f"region={region}" / f"year={year}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            # Use UUID for filename to support massive parallel ingestion (append-only)
            file_name = f"{uuid.uuid4()}.parquet"
            file_path = partition_dir / file_name

            # Drop the partition column 'year' before saving, as it's in the directory structure
            # Keep 'symbol' as it's a column, not a directory partition in this level
            group_to_save = group.drop(columns=['year'])
            group_to_save.to_parquet(file_path, compression='snappy')

        logger.info(f"Ingested {symbol} into Hive partitions.")

    def load_data(self, symbol: str, region: str = None) -> pd.DataFrame:
        """
        Loads historical data for a symbol from the Lakehouse.
        """
        # Best effort load using pyarrow dataset filters if available via pandas
        try:
            return pd.read_parquet(
                self.daily_path,
                filters=[('symbol', '=', symbol)]
            )
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise

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
