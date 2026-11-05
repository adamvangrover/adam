import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class PhysicsValidationError(Exception):
    """Raised when market metrics violate physical bounds (e.g., negative VIX or excessive S&P variance)."""
    pass

class SyntheticMalwareError(Exception):
    """Raised when Whalescanner SEC filing target company does not mathematically match official CIK records."""
    pass

class BoundedMarketMetrics(BaseModel):
    sp500: float = Field(..., description="S&P 500 Index Level")
    yield_10y: float = Field(..., description="10-Year Treasury Yield Percentage")
    vix: float = Field(..., description="VIX Volatility Index")
    btc: float = Field(..., description="Bitcoin Price USD")
    brent: float = Field(..., description="Brent Crude Oil Price USD")
    sp500_daily_variance_pct: float = Field(0.0, description="Daily percentage variance of S&P 500")

    @field_validator("vix")
    @classmethod
    def validate_vix_bounds(cls, v: float) -> float:
        if v < 0.0:
            raise PhysicsValidationError(f"Physics violation: VIX cannot be negative ({v} < 0)")
        if v > 200.0:
            raise PhysicsValidationError(f"Physics violation: VIX exceeds theoretical maximum physical bound ({v} > 200)")
        return v

    @field_validator("sp500_daily_variance_pct")
    @classmethod
    def validate_sp500_variance(cls, v: float) -> float:
        if abs(v) > 25.0:
            raise PhysicsValidationError(f"Physics violation: S&P 500 daily variance exceeds 25% physical limit ({v}%)")
        return v

    @field_validator("sp500", "yield_10y", "btc", "brent")
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        if v < 0.0:
            raise PhysicsValidationError(f"Physics violation: {info.field_name} cannot be negative ({v})")
        return v


# SEC CIK Database Mock / Registry for Whalescanner Cross-Referencing
MOCK_SEC_CIK_DATABASE = {
    "0000320193": {"ticker": "AAPL", "name": "Apple Inc.", "shares_outstanding": 15200000000},
    "0000789019": {"ticker": "MSFT", "name": "Microsoft Corp", "shares_outstanding": 7430000000},
    "0001652044": {"ticker": "GOOGL", "name": "Alphabet Inc.", "shares_outstanding": 12500000000},
    "0001018724": {"ticker": "AMZN", "name": "Amazon.com Inc.", "shares_outstanding": 10400000000},
    "0001318605": {"ticker": "TSLA", "name": "Tesla Inc.", "shares_outstanding": 3180000000},
    "000050863":  {"ticker": "INTC", "name": "Intel Corp", "shares_outstanding": 4260000000},
}

class WhalescannerFiling(BaseModel):
    filing_type: str = Field(..., description="13D, 13F, or 14A")
    target_cik: str = Field(..., description="Official SEC CIK 10-digit zero-padded code")
    target_ticker: str
    reported_shares: int
    implied_ownership_pct: float

class WhalescannerValidator:
    """
    Whalescanner Validation Hook:
    Cross-references ingested 13D/13F/14A target company against the official SEC CIK database.
    If target entity does not mathematically align with its outstanding share count,
    flags payload as synthetic malware and raises SyntheticMalwareError.
    """
    def __init__(self, cik_db: Optional[Dict[str, Dict[str, Any]]] = None):
        self.cik_db = cik_db or MOCK_SEC_CIK_DATABASE

    def validate_filing(self, filing: WhalescannerFiling) -> bool:
        padded_cik = filing.target_cik.zfill(10)

        if padded_cik not in self.cik_db:
            # Unknown or hallucinated CIK
            raise SyntheticMalwareError(f"Whalescanner Malware Rejection: CIK '{filing.target_cik}' not found in SEC database")

        official_record = self.cik_db[padded_cik]
        total_shares = official_record["shares_outstanding"]

        # Verify mathematical alignment of reported shares and ownership percentage
        expected_pct = (filing.reported_shares / total_shares) * 100.0
        pct_diff = abs(expected_pct - filing.implied_ownership_pct)

        if pct_diff > 1.0: # Discrepancy > 1% indicates hallucinated/synthetic filing data
            raise SyntheticMalwareError(
                f"Whalescanner Synthetic Malware Injection Detected! CIK {padded_cik} ({official_record['ticker']}): "
                f"Reported shares {filing.reported_shares} yields {expected_pct:.2f}% ownership, "
                f"which mathematically violates reported {filing.implied_ownership_pct:.2f}%"
            )

        if filing.reported_shares > total_shares:
            raise SyntheticMalwareError(
                f"Whalescanner Synthetic Malware Injection Detected! "
                f"Reported shares ({filing.reported_shares}) exceeds total outstanding shares ({total_shares})"
            )

        return True
