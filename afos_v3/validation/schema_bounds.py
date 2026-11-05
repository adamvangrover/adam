"""
AFOS V3 Deterministic Schema & Integrity Validation Layer
Phase 3: Immutable Physical Bounds & Whalescanner SEC CIK Cross-Referencing
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from afos_v3.gateway.zero_trust_gateway import QuarantineSink, ZeroTrustSecurityError

MAX_SP500_DAILY_VARIANCE: Final[float] = 0.25  # 25% hard physical limit
MAX_OWNERSHIP_ALIGNMENT_TOLERANCE_PCT: Final[float] = 0.05  # 5 bps mathematical tolerance
CIK_REGEX: Final[re.Pattern[str]] = re.compile(r"^\d{10}$")


class KernelPhysicsViolationError(ZeroTrustSecurityError):
    """Raised when market telemetry violates immutable physical bounds."""


class SyntheticMalwareFilingError(ZeroTrustSecurityError):
    """Raised when a 13D/13F/14A filing fails SEC CIK or share-count mathematical verification."""


class CoreMetricsPayload(BaseModel):
    """
    Phase 3.1: Strict runtime schema and physical boundary enforcement
    for Core Metrics (S&P 500, 10Y Yield, VIX, BTC, Brent).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    timestamp_epoch: float = Field(..., gt=0.0)
    sp500_price: float = Field(..., gt=0.0, le=100_000.0)
    sp500_prev_close: float = Field(..., gt=0.0, le=100_000.0)
    us10y_yield: float = Field(..., ge=-5.0, le=30.0)
    vix: float = Field(..., ge=0.0, le=200.0)
    btc_usd: float = Field(..., gt=0.0, le=10_000_000.0)
    brent_crude: float = Field(..., gt=0.0, le=1_000.0)

    @field_validator("sp500_price", "sp500_prev_close", "us10y_yield", "vix", "btc_usd", "brent_crude")
    @classmethod
    def reject_non_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Non-finite floating point value (NaN/Inf) prohibited.")
        return value

    @model_validator(mode="after")
    def enforce_physical_market_bounds(self) -> CoreMetricsPayload:
        daily_variance = abs(self.sp500_price - self.sp500_prev_close) / self.sp500_prev_close
        if daily_variance > MAX_SP500_DAILY_VARIANCE:
            raise ValueError(
                f"S&P 500 daily variance ({daily_variance:.2%}) exceeds immutable physical ceiling "
                f"({MAX_SP500_DAILY_VARIANCE:.0%})."
            )
        return self


@dataclass(frozen=True, slots=True)
class SECCIKEntityRecord:
    """Verified authoritative SEC EDGAR issuer record."""

    cik: str
    ticker: str
    legal_name: str
    shares_outstanding: int


class WhalescannerFilingPayload(BaseModel):
    """Schema for incoming SEC 13D / 13F / 14A filings ingested by Whalescanner."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    accession_number: str = Field(..., min_length=18, max_length=25)
    form_type: Literal["SC 13D", "SC 13D/A", "13F-HR", "DEF 14A"]
    filer_cik: str
    target_cik: str
    target_ticker: str
    reported_shares_held: int = Field(..., gt=0)
    reported_ownership_pct: float = Field(..., gt=0.0, le=100.0)

    @field_validator("filer_cik", "target_cik")
    @classmethod
    def validate_cik_format(cls, value: str) -> str:
        if not CIK_REGEX.match(value):
            raise ValueError(f"CIK '{value}' must be a strict 10-digit zero-padded string.")
        return value


class DeterministicValidationEngine:
    """
    Phase 3 Validation Engine combining CoreMetrics physical bounds checking
    and Whalescanner SEC CIK cross-referencing with automatic QuarantineSink routing.
    """

    def __init__(
        self,
        quarantine_sink: QuarantineSink,
        sec_cik_registry: dict[str, SECCIKEntityRecord] | None = None,
    ) -> None:
        self._quarantine = quarantine_sink
        self._sec_registry: dict[str, SECCIKEntityRecord] = sec_cik_registry or {}

    def register_sec_entity(self, record: SECCIKEntityRecord) -> None:
        self._sec_registry[record.cik] = record

    async def validate_core_metrics(
        self,
        node_id: str,
        raw_data: dict[str, Any],
    ) -> CoreMetricsPayload:
        try:
            return CoreMetricsPayload.model_validate(raw_data)
        except (ValidationError, ValueError) as exc:
            await self._quarantine.dump(
                failure_category="KERNEL_PHYSICS_VIOLATION",
                source_node=node_id,
                reason=str(exc),
                raw_payload=raw_data,
            )
            raise KernelPhysicsViolationError(
                f"Kernel rejected payload from '{node_id}' due to physical/schema violation: {exc}"
            ) from exc

    async def validate_whalescanner_filing(
        self,
        node_id: str,
        raw_filing: dict[str, Any],
    ) -> WhalescannerFilingPayload:
        try:
            filing = WhalescannerFilingPayload.model_validate(raw_filing)
        except (ValidationError, ValueError) as exc:
            await self._quarantine.dump(
                failure_category="SYNTHETIC_MALWARE_FILING_SCHEMA",
                source_node=node_id,
                reason=str(exc),
                raw_payload=raw_filing,
            )
            raise SyntheticMalwareFilingError(
                f"Whalescanner schema rejection from '{node_id}': {exc}"
            ) from exc

        # Cross-reference against official SEC CIK registry
        official_entity = self._sec_registry.get(filing.target_cik)
        if official_entity is None:
            reason = (
                f"Target CIK '{filing.target_cik}' ({filing.target_ticker}) does not exist in "
                "verified SEC EDGAR CIK registry. Flagged as synthetic malware."
            )
            await self._quarantine.dump(
                failure_category="SYNTHETIC_MALWARE_UNVERIFIED_CIK",
                source_node=node_id,
                reason=reason,
                raw_payload=raw_filing,
            )
            raise SyntheticMalwareFilingError(reason)

        if official_entity.ticker.upper() != filing.target_ticker.upper():
            reason = (
                f"Target CIK '{filing.target_cik}' ticker mismatch: filing claims "
                f"'{filing.target_ticker}', SEC registry specifies '{official_entity.ticker}'."
            )
            await self._quarantine.dump(
                failure_category="SYNTHETIC_MALWARE_TICKER_MISMATCH",
                source_node=node_id,
                reason=reason,
                raw_payload=raw_filing,
            )
            raise SyntheticMalwareFilingError(reason)

        # Mathematical alignment check against authoritative shares outstanding
        if filing.reported_shares_held > official_entity.shares_outstanding:
            reason = (
                f"Impossible share position on {official_entity.ticker}: reported "
                f"{filing.reported_shares_held:,} shares exceeds total outstanding shares "
                f"({official_entity.shares_outstanding:,}). Flagged as synthetic malware."
            )
            await self._quarantine.dump(
                failure_category="SYNTHETIC_MALWARE_SHARE_COUNT_OVERFLOW",
                source_node=node_id,
                reason=reason,
                raw_payload=raw_filing,
            )
            raise SyntheticMalwareFilingError(reason)

        implied_pct = (filing.reported_shares_held / official_entity.shares_outstanding) * 100.0
        pct_discrepancy = abs(implied_pct - filing.reported_ownership_pct)

        if pct_discrepancy > MAX_OWNERSHIP_ALIGNMENT_TOLERANCE_PCT:
            reason = (
                f"Mathematical share-count misalignment on {official_entity.ticker} ({filing.form_type}): "
                f"reported {filing.reported_shares_held:,} / {official_entity.shares_outstanding:,} "
                f"implies {implied_pct:.4f}% ownership, but filing claims "
                f"{filing.reported_ownership_pct:.4f}% (delta {pct_discrepancy:.4f}% > "
                f"{MAX_OWNERSHIP_ALIGNMENT_TOLERANCE_PCT:.4f}%)."
            )
            await self._quarantine.dump(
                failure_category="SYNTHETIC_MALWARE_MATH_MISALIGNMENT",
                source_node=node_id,
                reason=reason,
                raw_payload=raw_filing,
            )
            raise SyntheticMalwareFilingError(reason)

        return filing
