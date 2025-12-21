import os
from typing import List

import pandas as pd
import pandera as pa
import yfinance as yf

from core.schemas.market_data_schema import MarketDataSchema
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

class HistoricalLoader:
    """
    Handles the bulk ingestion, validation, and storage of historical market data.
    """

    def __init__(self, data_dir: str = "data/market_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_history(self, tickers: List[str], start_date: str = "1980-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
        """
        Fetches historical data for a list of tickers using yfinance bulk download.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Long-format DataFrame with [Date, Ticker, Open, High, Low, Close, Volume, ...]
        """
        logger.info(f"Fetching historical data for {len(tickers)} tickers from {start_date} to {end_date}...")

        if not tickers:
            logger.warning("No tickers provided.")
            return pd.DataFrame()

        try:
            # yfinance download
            # threads=True enables multi-threading
            # group_by='ticker' makes it easier to stack later, or 'column'
            # auto_adjust=False ensures we get 'Adj Close' separate or we can use auto_adjust=True for OHLC adjusted.
            # Usually raw OHLC + Adj Close is preferred.
            df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True, auto_adjust=False)

            if df.empty:
                logger.warning("Downloaded data is empty.")
                return pd.DataFrame()

            # Reshape to long format
            # If single ticker, yf returns simple DF. If multiple, MultiIndex.
            if len(tickers) == 1:
                df['Ticker'] = tickers[0]
                df = df.reset_index()
            else:
                # df columns are MultiIndex (Ticker, PriceType)
                # Stack to get Ticker as a column
                # The level 0 is Ticker, level 1 is PriceType (Open, Close, etc.)
                # We want Ticker as a column.
                # Stack level 0 (Ticker)
                df = df.stack(level=0)
                df.index.names = ['Date', 'Ticker']
                df = df.reset_index()

            logger.info(f"Fetched {len(df)} rows of data.")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the DataFrame against the MarketDataSchema.
        """
        logger.info("Validating data schema...")
        try:
            # Ensure proper types before validation
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])

            validated_df = MarketDataSchema.validate(df)
            logger.info("Data validation successful.")
            return validated_df
        except pa.errors.SchemaError as e:
            logger.error(f"Schema validation error: {e}")
            # Depending on policy, we might raise or return partial data.
            # For now, we return the dataframe but log the error (or we could drop bad rows).
            # Let's return the original df but log heavily.
            return df
        except Exception as e:
             logger.error(f"Validation failed: {e}")
             return df

    def save_to_parquet(self, df: pd.DataFrame, filename: str = "historical_data.parquet", compression: str = "snappy"):
        """
        Saves the DataFrame to a Parquet file.
        """
        if df.empty:
            logger.warning("DataFrame is empty, not saving.")
            return

        filepath = os.path.join(self.data_dir, filename)
        logger.info(f"Saving data to {filepath} with compression='{compression}'...")
        try:
            df.to_parquet(filepath, engine='pyarrow', compression=compression, index=False)
            logger.info(f"Successfully saved {os.path.getsize(filepath) / (1024*1024):.2f} MB to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to parquet: {e}")

    def run_pipeline(self, tickers: List[str], filename: str = "historical_market_data.parquet"):
        """
        Runs the full fetch-validate-save pipeline.
        """
        df = self.fetch_history(tickers)
        if not df.empty:
            df = self.validate_data(df)
            self.save_to_parquet(df, filename=filename)
        else:
            logger.error("Pipeline failed: No data fetched.")
