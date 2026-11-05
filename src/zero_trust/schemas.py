"""
Deterministic Schemas with Rigid Physical Bounds.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo


class MarketMetricsPayload(BaseModel):
    """
    Pydantic schema enforcing rigid physical bounds on market metrics.
    """
    vix: float = Field(..., description="CBOE Volatility Index")
    sp500_daily_variance: float = Field(..., description="S&P 500 Daily Variance percentage")
    sp500_index: float = Field(..., description="S&P 500 Index Level")
    treasury_10y_yield: float = Field(..., description="10-Year Treasury Yield percentage")
    btc_usd: float = Field(..., description="Bitcoin USD Price")
    brent_crude: float = Field(..., description="Brent Crude Price USD")

    @field_validator("vix")
    @classmethod
    def validate_vix_bounds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("VIX physics violation: VIX cannot be negative (< 0).")
        if v > 200:
            raise ValueError("VIX physics violation: VIX > 200 exceeds physical limits.")
        return v

    @field_validator("sp500_daily_variance")
    @classmethod
    def validate_sp500_variance(cls, v: float) -> float:
        if abs(v) > 25.0:
            raise ValueError(f"S&P 500 daily variance violation: |{v}%| exceeds rigid limit of 25%.")
        return v

    @field_validator("treasury_10y_yield")
    @classmethod
    def validate_10y_yield(cls, v: float) -> float:
        if v < -2.0 or v > 30.0:
            raise ValueError(f"10Y Treasury yield violation: {v}% outside physical bound (-2% to 30%).")
        return v

    @field_validator("brent_crude", "btc_usd", "sp500_index")
    @classmethod
    def validate_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} physics violation: value cannot be negative ({v}).")
        return v


class SECFilingIngestionPayload(BaseModel):
    """
    Schema for SEC Whalescanner filings (13D, 13F, 14A).
    """
    filing_type: str = Field(..., description="SEC Form Type e.g., 13D, 13F, 14A")
    target_cik: str = Field(..., description="Target Company CIK (10 digits)")
    target_ticker: str = Field(..., description="Target Ticker Symbol")
    shares_reported: int = Field(..., description="Reported share ownership count")
    total_shares_outstanding: int = Field(..., description="Official outstanding share count")

    @field_validator("filing_type")
    @classmethod
    def validate_form_type(cls, v: str) -> str:
        valid_types = ["13D", "13F", "14A", "FORM 13D", "FORM 13F", "FORM 14A"]
        if v.upper() not in valid_types:
            raise ValueError(f"Unsupported SEC filing type: {v}")
        return v.upper()

    @field_validator("target_cik")
    @classmethod
    def validate_cik_format(cls, v: str) -> str:
        digits_only = v.lstrip("0")
        if not digits_only.isdigit():
            raise ValueError(f"Invalid CIK format: {v}")
        return v.zfill(10)
