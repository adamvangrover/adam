import abc
import logging
import uuid
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class RealTimePipe(abc.ABC):
    """
    Abstract base class for real-time data ingestion.
    """

    def __init__(self, lakehouse_path: str = "data/daily", flush_threshold: int = 100):
        self.lakehouse_path = Path(lakehouse_path)
        self.buffer: List[Dict[str, Any]] = []
        self.flush_threshold = flush_threshold

    @abc.abstractmethod
    def listen(self):
        """Starts listening to the data stream."""
        pass

    def push_tick(self, tick: Dict[str, Any]):
        """Buffers a single tick and flushes if threshold reached."""
        self.buffer.append(tick)
        if len(self.buffer) >= self.flush_threshold:
            self.flush()

    def flush(self):
        """Writes buffered data to the lakehouse."""
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        # Ensure timestamp
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp in buffer, skipping flush.")
            self.buffer = []
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Partitioning logic
        if 'symbol' not in df.columns:
            logger.warning("No symbol in buffer, skipping flush.")
            self.buffer = []
            return

        # Heuristic for region if not present
        if 'region' not in df.columns:
            df['region'] = df['symbol'].apply(self._get_region)

        df['year'] = df.index.year

        # Group by partition keys to write appropriate files
        for (region, year), group in df.groupby(['region', 'year']):
            partition_dir = self.lakehouse_path / f"region={region}" / f"year={year}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"{uuid.uuid4()}.parquet"
            file_path = partition_dir / file_name

            # Clean up columns: remove partition keys if they are not needed in the file
            # Usually in Hive style, they are implicit, but keeping them doesn't hurt.
            # Lakehouse implementation dropped 'year' but kept 'symbol'.
            # We will drop 'year' and 'region'.
            group_to_save = group.drop(columns=['year', 'region'])

            try:
                group_to_save.to_parquet(file_path, compression='snappy')
                logger.info(f"Flushed {len(group)} records to {file_path}")
            except Exception as e:
                logger.error(f"Failed to flush to {file_path}: {e}")

        self.buffer = []

    def _get_region(self, symbol: str) -> str:
        """Determines region from ticker symbol."""
        if "." not in symbol:
            return "US"
        suffix = symbol.split(".")[-1]
        if suffix in ["L", "PA", "DE", "AM", "BR", "MI", "MC", "LS", "IR", "AS", "HE", "SW"]:
            return "EU"
        if suffix in ["HK", "T", "SS", "SZ", "KS", "AX", "NZ"]:
            return "APAC"
        return "ROW"


class MockRealTimePipe(RealTimePipe):
    """
    Simulates a real-time feed for testing.
    """

    def listen(self):
        # In a real scenario, this would loop forever.
        import random
        from datetime import timedelta

        symbols = ['AAPL', 'MSFT', 'VOD.L', 'SIE.DE']
        base_time = datetime.now()

        logger.info("Starting Mock Stream...")

        # Generate 2 flushes worth of data
        for i in range(self.flush_threshold * 2 + 10):
            symbol = random.choice(symbols)
            price = random.uniform(100, 200)

            # Late arrival simulation: 10% chance of being yesterday
            if random.random() < 0.1:
                timestamp = base_time - timedelta(days=1)
            else:
                timestamp = base_time + timedelta(minutes=i)

            tick = {
                "symbol": symbol,
                "timestamp": timestamp,
                "price": price,
                "volume": random.randint(100, 1000)
            }
            self.push_tick(tick)

        self.flush()  # Flush remaining
