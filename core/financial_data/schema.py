from __future__ import annotations
from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class MarketTicker(BaseModel):
    """
    Represents a financial instrument discovered in the market.
    Aligns with 'Path A' strict typing requirements.
    """
    symbol: str = Field(..., description="The ticker symbol (e.g., AAPL)")
    short_name: Optional[str] = Field(None, alias="shortname", description="Short display name")
    long_name: Optional[str] = Field(None, alias="longname", description="Full company name")
    exchange: Optional[str] = Field(None, description="Exchange code (e.g., NMS)")
    quote_type: Optional[str] = Field(None, alias="quoteType", description="Type of asset (EQUITY, ETF, etc.)")
    sector: Optional[str] = Field(None, description="Industry sector")
    industry: Optional[str] = Field(None, description="Specific industry")
    score: Optional[float] = Field(None, description="Search relevance score")

    # Metadata for ingestion tracking
    discovery_date: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)

    model_config = ConfigDict(populate_by_name=True)

class TickerList(BaseModel):
    tickers: List[MarketTicker]

class HistoricalPrice(BaseModel):
    """
    Represents a single candle of price data.
    """
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    dividends: float
    stock_splits: float

    model_config = ConfigDict(populate_by_name=True)
