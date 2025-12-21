"""
Ingestion Engine for the Gold Standard Toolkit.
Handles reliable data fetching from yfinance with rate limit management.
"""

import logging
import time
from typing import Any, Dict, List

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from .qa import validate_dataframe
from .storage import StorageEngine

logger = logging.getLogger(__name__)

class IngestionEngine:
    def __init__(self, storage: StorageEngine):
        self.storage = storage

    def _download_chunk(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """
        Downloads a chunk of tickers with exponential backoff.
        """
        max_retries = 3
        delay = 2

        for attempt in range(max_retries):
            try:
                # yf.download returns a MultiIndex DataFrame if len(tickers) > 1
                data = yf.download(tickers, **kwargs)
                return data
            except Exception as e:
                logger.warning(f"Download failed (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

        logger.error(f"Failed to download chunk: {tickers}")
        return pd.DataFrame()

    def ingest_daily_history(self, tickers: List[str], period: str = "max", interval: str = "1d"):
        """
        Batch downloads daily history and stores it using Time-Based Partitioning.
        """
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            logger.info(f"Ingesting daily batch: {chunk}")

            # Download
            df = self._download_chunk(chunk, period=period, interval=interval, group_by='ticker', auto_adjust=True)

            if df.empty:
                continue

            # yfinance group_by='ticker' returns DataFrame with Ticker as top level column if multi,
            # but if single ticker, it's flat.
            # However, for Time-Based Partitioning (StorageEngine.store_daily), we ideally want a long format or specific schema.
            # The blueprint says "returns a Pandas DataFrame with a MultiIndex... efficient for memory but requires reshaping".

            # Let's standardize to Long Format for storage: Date, Ticker, Open, High...
            # If multi-index columns (Ticker, Price):
            if isinstance(df.columns, pd.MultiIndex):
                # Stack to get Ticker as a column (or index level)
                # Current: Columns=(Ticker, OHLCV), Index=Date
                # We want: Index=Date, Columns=OHLCV, and Ticker as a column?
                # Or StorageEngine expects a dataframe with many tickers?
                # StorageEngine.store_daily expects a DF that it will partition by YEAR.
                # If we store wide format (Cols = Ticker_Open, Ticker_Close), parquet handles it but schema is huge.
                # Usually Long format is better: Date, Ticker, Open...

                # Let's stack.
                df_long = df.stack(level=0) # Moves Ticker to index
                df_long.index.names = ['Date', 'Ticker']
                df_long = df_long.reset_index(level='Ticker') # Ticker is now a column

                # Validate
                try:
                    validate_dataframe(df_long)
                except Exception as e:
                    logger.error(f"Validation failed for batch: {e}. Storing anyway.")
                self.storage.store_daily(df_long)
            else:
                # Single ticker or flat structure
                # We assume list of tickers > 1 usually. If 1, handle it.
                if len(chunk) == 1:
                    ticker = chunk[0]
                    df['Ticker'] = ticker
                    try:
                        validate_dataframe(df)
                    except Exception as e:
                        logger.error(f"Validation failed for {ticker}: {e}. Storing anyway.")
                    self.storage.store_daily(df)

    def ingest_intraday_eager(self, tickers: List[str]):
        """
        Eagerly ingests 1m data for the last 7 days.
        Must be run weekly to prevent data loss.
        """
        logger.info("Starting eager intraday ingestion...")
        for ticker in tickers:
            try:
                # fetch 1m data, prepost=True for extended hours
                df = self._download_chunk([ticker], period="7d", interval="1m", prepost=True)

                if not df.empty:
                    # Flatten MultiIndex if present (yfinance returns MultiIndex if input is list)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # Ensure index name is Date for schema validation
                    df.index.name = "Date"

                    # Convert index to naive datetime to satisfy Pandera schema (and avoid TZ issues)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    # Validate
                    try:
                        validate_dataframe(df)
                    except Exception as val_e:
                        logger.error(f"Validation failed for {ticker}: {val_e}. Storing anyway.")
                        logger.error(f"Head: {df.head()}")
                    # Store
                    self.storage.store_intraday(df, ticker, frequency="1m")
            except Exception as e:
                logger.error(f"Intraday ingestion failed for {ticker}: {e}")

    def get_realtime_snapshot(self, ticker: str) -> Dict[str, Any]:
        """
        Uses Ticker.fast_info for low-latency L1 data.
        """
        if yf is None: return {}

        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info

            # fast_info attributes
            snapshot = {
                "symbol": ticker,
                "last_price": fi.last_price,
                "previous_close": fi.previous_close,
                "open": fi.open,
                "day_high": fi.day_high,
                "day_low": fi.day_low,
                "year_high": fi.year_high,
                "year_low": fi.year_low,
                "market_cap": fi.market_cap,
                "currency": fi.currency,
                "timestamp": time.time()
            }
            return snapshot
        except Exception as e:
            logger.error(f"Snapshot failed for {ticker}: {e}")
            return {}
