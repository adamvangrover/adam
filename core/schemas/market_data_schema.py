import pandera as pa
from pandera.typing import Series


class MarketDataSchema(pa.DataFrameModel):
    """
    Pandera schema for validating historical market data.
    Expected format: Long format with Date and Ticker as columns or index.
    """
    Date: Series[pa.DateTime] = pa.Field(coerce=True)
    Ticker: Series[pa.String] = pa.Field()
    Open: Series[pa.Float] = pa.Field(ge=0, nullable=True)
    High: Series[pa.Float] = pa.Field(ge=0, nullable=True)
    Low: Series[pa.Float] = pa.Field(ge=0, nullable=True)
    Close: Series[pa.Float] = pa.Field(ge=0, nullable=True)
    Volume: Series[pa.Float] = pa.Field(ge=0, nullable=True)  # Float because sometimes it's fractional or huge

    class Config:
        strict = False # Allow extra columns like 'Adj Close'
        coerce = True
