"""
SEC CIK Cross-Referencing & Share Count Alignment Validation Hook.
Cross-references 13D/13F/14A target companies against official SEC CIK database.
Flags payloads failing share count alignment or CIK verification as synthetic malware.
"""

from typing import Dict, Any, Optional
from pydantic import ValidationError
from .schemas import SECFilingIngestionPayload


class SECIntegrityError(ValueError):
    """Raised when an SEC filing payload fails CIK cross-referencing or share count physics."""
    pass


# Mock SEC EDGAR Official CIK & Share Count Database
OFFICIAL_SEC_CIK_REGISTRY = {
    "0000789019": {"ticker": "MSFT", "name": "MICROSOFT CORP", "total_outstanding_shares": 7430000000},
    "0000320193": {"ticker": "AAPL", "name": "APPLE INC", "total_outstanding_shares": 15400000000},
    "0001018724": {"ticker": "AMZN", "name": "AMAZON COM INC", "total_outstanding_shares": 10400000000},
    "0001652044": {"ticker": "GOOGL", "name": "ALPHABET INC", "total_outstanding_shares": 12300000000},
    "0001326801": {"ticker": "META", "name": "META PLATFORMS INC", "total_outstanding_shares": 2540000000},
}


class WhalescannerSECValidator:
    """
    Validates 13D/13F/14A filings by cross-referencing target CIK and outstanding share alignment.
    """

    def __init__(self, cik_registry: Optional[Dict[str, Dict[str, Any]]] = None):
        self.cik_registry = cik_registry or OFFICIAL_SEC_CIK_REGISTRY

    def validate_filing_payload(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates SEC filing payload.
        Raises SECIntegrityError if target entity does not mathematically align with CIK database.
        """
        try:
            payload = SECFilingIngestionPayload(**raw_payload)
        except ValidationError as e:
            raise SECIntegrityError(f"SEC Filing Schema Validation Failed: {e}")

        cik = payload.target_cik
        reg_entry = self.cik_registry.get(cik)

        # 1. CIK Existence Check
        if not reg_entry:
            raise SECIntegrityError(
                f"Synthetic Malware Alert! CIK {cik} for ticker {payload.target_ticker} not found in official SEC CIK database."
            )

        # 2. Ticker Cross-Reference Check
        expected_ticker = reg_entry["ticker"]
        if payload.target_ticker.upper() != expected_ticker:
            raise SECIntegrityError(
                f"Synthetic Malware Alert! Mismatch between CIK {cik} ({reg_entry['name']}, ticker: {expected_ticker}) and reported ticker {payload.target_ticker}."
            )

        # 3. Share Count Mathematical Alignment
        official_outstanding = reg_entry["total_outstanding_shares"]

        if payload.shares_reported > official_outstanding:
            raise SECIntegrityError(
                f"Synthetic Malware Alert! Reported shares ({payload.shares_reported:,}) exceed official outstanding shares ({official_outstanding:,}) for CIK {cik}."
            )

        # Discrepancy in reported total shares vs SEC official record
        share_diff = abs(payload.total_shares_outstanding - official_outstanding)
        discrepancy_pct = share_diff / official_outstanding

        if discrepancy_pct > 0.10:  # > 10% discrepancy in reported total shares
            raise SECIntegrityError(
                f"Synthetic Malware Alert! Reported total shares ({payload.total_shares_outstanding:,}) deviates by {discrepancy_pct:.2%} from SEC official record ({official_outstanding:,}). Hallucinated filing detected."
            )

        return {
            "verified": True,
            "cik": cik,
            "company_name": reg_entry["name"],
            "ticker": expected_ticker,
            "filing_type": payload.filing_type,
            "shares_reported": payload.shares_reported,
            "ownership_pct": round((payload.shares_reported / official_outstanding) * 100, 4)
        }
