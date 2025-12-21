"""
Storage Layer for the Gold Standard Toolkit.
Implements Data Lakehouse architecture using Apache Parquet and PyArrow.
"""

import logging
import os
import uuid
from typing import Optional

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

logger = logging.getLogger(__name__)

class StorageEngine:
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        if pa is None:
            logger.warning("PyArrow is not installed. Storage operations will fail.")

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def store_intraday(self, df: pd.DataFrame, ticker: str, frequency: str = "1m"):
        """
        Stores intraday data partitioned by Ticker and Year.
        Strategy: Ticker-Based Partitioning.
        """
        if df.empty:
            logger.warning(f"No data to store for {ticker}")
            return

        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # Add partition column
        df = df.copy()
        df['year'] = df.index.year

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Define path: data/intraday/frequency=1m/ticker=TSLA/
        output_path = os.path.join(self.base_path, "intraday", f"frequency={frequency}", f"ticker={ticker}")

        # Write to dataset (Partition by year)
        # basename_template ensures we append new files instead of overwriting existing ones (UUID)
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=['year'],
            basename_template=f"part-{{i}}-{uuid.uuid4()}.parquet",
            existing_data_behavior='overwrite_or_ignore'
        )
        logger.info(f"Stored intraday data for {ticker} at {output_path}")

    def store_daily(self, df: pd.DataFrame, region: str = "US"):
        """
        Stores daily/long-term data partitioned by Region and Year.
        Strategy: Time-Based Partitioning (optimized for cross-sectional queries).
        """
        if df.empty:
            logger.warning("No daily data to store.")
            return

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        df = df.copy()
        df['year'] = df.index.year

        table = pa.Table.from_pandas(df)

        output_path = os.path.join(self.base_path, "daily", f"region={region}")

        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=['year'],
            existing_data_behavior='overwrite_or_ignore'
        )
        logger.info(f"Stored daily data for region {region} at {output_path}")

    def load_intraday(self, ticker: str, frequency: str = "1m",
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Loads intraday data with Predicate Pushdown optimization.
        """
        path = os.path.join(self.base_path, "intraday", f"frequency={frequency}", f"ticker={ticker}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"No data found for {ticker} at {path}")

        # PyArrow dataset scanning
        dataset = pq.ParquetDataset(path)

        # We can implement filter pushdown here if needed, but basic reading is:
        table = dataset.read()
        df = table.to_pandas()

        # Sort index as multiple files might be read out of order
        df = df.sort_index()

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def load_daily(self, region: str = "US", year: Optional[int] = None) -> pd.DataFrame:
        """
        Loads daily data for a whole region (or specific year).
        """
        path = os.path.join(self.base_path, "daily", f"region={region}")

        filters = None
        if year:
            filters = [('year', '=', year)]

        dataset = pq.ParquetDataset(path, filters=filters)
        table = dataset.read()
        df = table.to_pandas()
        return df.sort_index()
