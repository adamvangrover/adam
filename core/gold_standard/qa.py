"""
Quality Assurance Module for the Gold Standard Toolkit.
Handles schema validation using Pandera and market calendar logic.
"""

import logging
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
except ImportError:
    pa = None
    Column = Check = DataFrameSchema = object  # Mock for linting if missing

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

logger = logging.getLogger(__name__)

# --- Schema Definitions ---


def get_market_data_schema() -> 'DataFrameSchema':
    if pa is None:
        raise ImportError("Pandera is required for schema validation.")

    return DataFrameSchema({
        "Open": Column(float, Check.ge(0), nullable=True),
        "High": Column(float, Check.ge(0), nullable=True),
        "Low": Column(float, Check.ge(0), nullable=True),
        "Close": Column(float, Check.ge(0), nullable=True),
        "Volume": Column(float, Check.ge(0), nullable=True, coerce=True),
        "Adj Close": Column(float, Check.ge(0), nullable=True, required=False),
        "Dividends": Column(float, Check.ge(0), nullable=True, required=False),
        "Stock Splits": Column(float, Check.ge(0), nullable=True, required=False),
    }, index=pa.Index(pa.DateTime, name="Date"))


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the dataframe against the Gold Standard schema.
    """
    schema = get_market_data_schema()
    try:
        # Enforce index to be datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        return schema.validate(df)
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise e

# --- Market Calendar Logic ---


def is_market_holiday(date: pd.Timestamp, exchange: str = 'NYSE') -> bool:
    """
    Checks if a given date is a market holiday.
    """
    if mcal is None:
        logger.warning("pandas_market_calendars not installed. Assuming not a holiday.")
        return False

    nyse = mcal.get_calendar(exchange)
    schedule = nyse.schedule(start_date=date, end_date=date)
    return schedule.empty


def get_expected_market_days(start_date: str, end_date: str, exchange: str = 'NYSE') -> pd.DatetimeIndex:
    """
    Returns a list of expected trading days between start and end.
    """
    if mcal is None:
        return pd.date_range(start=start_date, end=end_date, freq='B')  # Fallback to business days

    nyse = mcal.get_calendar(exchange)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index
