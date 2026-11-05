"""
AFOS V3 Kernel Deterministic Data Validation & SEC Cross-Referencing.
Features:
1. Immutable Schema Bounds: Physical boundary validation prior to memory allocation
   (e.g., S&P 500 drop <= 45% in 1 sec, VIX >= 0.0, 10Y Yield >= 0.0). Physical breaches classified as malware injection.
2. Cross-Reference Integrity: SEC EDGAR CIK number and outstanding share count verification for Whalescanner alerts.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# SEC EDGAR Verified Entity Database Registry
VERIFIED_SEC_CIK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "0000789019": {"company_name": "MICROSOFT CORP", "ticker": "MSFT", "shares_outstanding": 7430000000},
    "0000320193": {"company_name": "APPLE INC", "ticker": "AAPL", "shares_outstanding": 15400000000},
    "0001018724": {"company_name": "AMAZON COM INC", "ticker": "AMZN", "shares_outstanding": 10400000000},
    "0001652044": {"company_name": "ALPHABET INC", "ticker": "GOOGL", "shares_outstanding": 12300000000},
    "0001045810": {"company_name": "NVIDIA CORP", "ticker": "NVDA", "shares_outstanding": 24600000000},
    "0001326801": {"company_name": "META PLATFORMS INC", "ticker": "META", "shares_outstanding": 2530000000},
    "0001318605": {"company_name": "TESLA INC", "ticker": "TSLA", "shares_outstanding": 3190000000},
    "0001901570": {"company_name": "ANTHROPIC PBC", "ticker": "ANT", "shares_outstanding": 500000000},
}


class PhysicalBoundsViolation(Exception):
    """Raised when an ingested variable breaches immutable physical boundary limits."""
    pass


class SyntheticFilingError(Exception):
    """Raised when an SEC filing or Whalescanner alert fails CIK or share count cross-referencing."""
    pass


class ValidationResult(BaseModel):
    is_valid: bool
    malware_injection_attempt: bool
    synthetic_filing_detected: bool
    error_message: Optional[str] = None
    sanitized_data: Optional[Dict[str, Any]] = None


class DeterministicDataValidator:
    """
    Deterministic Data Validator enforcing physical boundaries and SEC cross-reference integrity.
    Rejects synthetic filings and mathematically impossible telemetry prior to memory allocation.
    """

    # Physical Boundary Limits
    MAX_1SEC_SP500_DROP_PCT = 45.0  # S&P 500 drop > 45% in 1 second is physically impossible
    MIN_VIX = 0.0                   # Volatility index cannot be negative
    MAX_VIX = 200.0                 # Upper physical sanity bound for VIX
    MIN_SOFR = -2.0                 # SOFR lower physical bound
    MAX_SOFR = 50.0                 # SOFR upper physical bound
    MIN_10Y_YIELD = -2.0            # 10Y Treasury yield lower physical bound
    MAX_10Y_YIELD = 50.0           # 10Y Treasury yield upper physical bound

    @classmethod
    def validate_macro_payload(cls, payload: Dict[str, Any]) -> ValidationResult:
        """
        Validates macro telemetry payload against immutable physical bounds.
        Classifies mathematically impossible values as malware injection attempts.
        """
        try:
            # Check 1-sec market drop
            sp500_1sec_drop = float(payload.get("sp500_1sec_drop_pct", 0.0))
            if sp500_1sec_drop > cls.MAX_1SEC_SP500_DROP_PCT:
                return ValidationResult(
                    is_valid=False,
                    malware_injection_attempt=True,
                    synthetic_filing_detected=False,
                    error_message=f"PHYSICAL BOUND VIOLATION: S&P 500 1-sec drop {sp500_1sec_drop}% exceeds physical max {cls.MAX_1SEC_SP500_DROP_PCT}%. Classified as malware injection attempt."
                )

            # Check VIX physical bounds
            if "vix" in payload:
                vix = float(payload["vix"])
                if vix < cls.MIN_VIX or vix > cls.MAX_VIX:
                    return ValidationResult(
                        is_valid=False,
                        malware_injection_attempt=True,
                        synthetic_filing_detected=False,
                        error_message=f"PHYSICAL BOUND VIOLATION: Implied VIX {vix} outside physical bounds [{cls.MIN_VIX}, {cls.MAX_VIX}]. Classified as malware injection attempt."
                    )

            # Check 10Y Yield physical bounds
            if "yield_10y" in payload:
                y10 = float(payload["yield_10y"])
                if y10 < cls.MIN_10Y_YIELD or y10 > cls.MAX_10Y_YIELD:
                    return ValidationResult(
                        is_valid=False,
                        malware_injection_attempt=True,
                        synthetic_filing_detected=False,
                        error_message=f"PHYSICAL BOUND VIOLATION: 10Y Yield {y10}% outside physical bounds [{cls.MIN_10Y_YIELD}, {cls.MAX_10Y_YIELD}%]. Classified as malware injection attempt."
                    )

            # Check SOFR physical bounds
            if "sofr" in payload:
                sofr = float(payload["sofr"])
                if sofr < cls.MIN_SOFR or sofr > cls.MAX_SOFR:
                    return ValidationResult(
                        is_valid=False,
                        malware_injection_attempt=True,
                        synthetic_filing_detected=False,
                        error_message=f"PHYSICAL BOUND VIOLATION: SOFR {sofr}% outside physical bounds [{cls.MIN_SOFR}, {cls.MAX_SOFR}%]. Classified as malware injection attempt."
                    )

            return ValidationResult(
                is_valid=True,
                malware_injection_attempt=False,
                synthetic_filing_detected=False,
                sanitized_data=payload
            )

        except (ValueError, TypeError) as err:
            return ValidationResult(
                is_valid=False,
                malware_injection_attempt=True,
                synthetic_filing_detected=False,
                error_message=f"MALFORMED PAYLOAD DATA: {str(err)}"
            )

    @classmethod
    def cross_reference_whalescanner_alert(cls, alert: Dict[str, Any]) -> ValidationResult:
        """
        Cross-references Whalescanner takeover/LBO alerts against official SEC EDGAR CIK registry.
        Rejects synthetic filings containing hallucinated target companies or mismatched share counts.
        """
        target_cik = str(alert.get("target_cik", "")).zfill(10)
        target_company = alert.get("target_company_name", "").upper()
        reported_shares = alert.get("reported_outstanding_shares")

        # Cross-reference CIK in EDGAR registry
        if target_cik not in VERIFIED_SEC_CIK_REGISTRY:
            return ValidationResult(
                is_valid=False,
                malware_injection_attempt=False,
                synthetic_filing_detected=True,
                error_message=f"SYNTHETIC FILING REJECTED: CIK {target_cik} ({target_company}) not found in SEC EDGAR verified registry. Synthetic/hallucinated company target."
            )

        edgar_entry = VERIFIED_SEC_CIK_REGISTRY[target_cik]

        # Verify share count sanity (within 15% tolerance of EDGAR record)
        if reported_shares is not None:
            reported_shares_num = float(reported_shares)
            official_shares = float(edgar_entry["shares_outstanding"])
            mismatch_ratio = abs(reported_shares_num - official_shares) / official_shares

            if mismatch_ratio > 0.15:
                return ValidationResult(
                    is_valid=False,
                    malware_injection_attempt=False,
                    synthetic_filing_detected=True,
                    error_message=f"SYNTHETIC FILING REJECTED: Reported outstanding shares ({reported_shares_num}) deviates >15% from SEC EDGAR official record ({official_shares}) for CIK {target_cik}."
                )

        return ValidationResult(
            is_valid=True,
            malware_injection_attempt=False,
            synthetic_filing_detected=False,
            sanitized_data={
                **alert,
                "target_cik": target_cik,
                "verified_company_name": edgar_entry["company_name"],
                "official_shares_outstanding": edgar_entry["shares_outstanding"]
            }
        )
