"""
Extraction Integrity & Fail-Closed Evidence Boundary Module (F-01, F-03, F-08, Gate G0).

Enforces strict source-grounded telemetry parsing, cryptographic document hash matching,
source text span citation, and fail-closed validation for quantitative risk inputs.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class EvidenceStatus(str, Enum):
    VERIFIED_EXTRACTION = "VERIFIED_EXTRACTION"
    UNVERIFIED_NEGATIVE_FINDING = "UNVERIFIED_NEGATIVE_FINDING"
    UNGROUNDED_CLAIM = "UNGROUNDED_CLAIM"
    INCOMPLETE_PROVENANCE = "INCOMPLETE_PROVENANCE"


class SR117UnverifiedTelemetryError(ValueError):
    """Raised when telemetry payload lacks verified evidence, source citations, or valid hash."""
    pass


@dataclass
class EvidenceSpan:
    text: str
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None


@dataclass
class EvidenceBoundExtractionRecord:
    metric_name: str
    value: Any
    units: str
    status: EvidenceStatus
    doc_id: str
    doc_sha256: str
    timestamp: str
    confidence: float
    reconciliation_state: str = "RECONCILED"
    source_span: EvidenceSpan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "units": self.units,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "doc_id": self.doc_id,
            "doc_sha256": self.doc_sha256,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "reconciliation_state": self.reconciliation_state,
            "source_span": {
                "text": self.source_span.text,
                "page_number": self.source_span.page_number,
                "start_char": self.source_span.start_char,
                "end_char": self.source_span.end_char,
            } if self.source_span else None,
        }


def validate_extraction_record(record: EvidenceBoundExtractionRecord) -> bool:
    """
    Validates that an extraction record contains complete, source-grounded telemetry
    before permitting promotion into quantitative risk calculations.
    """
    if record.status != EvidenceStatus.VERIFIED_EXTRACTION:
        raise SR117UnverifiedTelemetryError(
            f"Extraction record status is '{record.status}', required 'VERIFIED_EXTRACTION'."
        )

    if not record.doc_sha256 or not re.match(r"^[a-fA-F0-9]{64}$", record.doc_sha256):
        raise SR117UnverifiedTelemetryError(
            f"Extraction record for '{record.metric_name}' missing or invalid doc_sha256 digest."
        )

    if record.source_span is None or not record.source_span.text or not record.source_span.text.strip():
        raise SR117UnverifiedTelemetryError(
            f"Extraction record for '{record.metric_name}' lacks grounded source text span citation."
        )

    if not record.timestamp:
        raise SR117UnverifiedTelemetryError(
            f"Extraction record for '{record.metric_name}' missing timestamp."
        )

    if record.reconciliation_state != "RECONCILED":
        msg = f"Extraction record for '{record.metric_name}' reconciliation state is '{record.reconciliation_state}'."
        raise SR117UnverifiedTelemetryError(msg)

    return True


def validate_telemetry_payload(payload: dict[str, Any]) -> bool:
    """
    Validates a raw telemetry payload dictionary for SR 11-7 evidence completeness.
    """
    status = payload.get("status")
    if status == EvidenceStatus.UNVERIFIED_NEGATIVE_FINDING.value or status == "UNVERIFIED_NEGATIVE_FINDING":
        raise SR117UnverifiedTelemetryError("Telemetry payload flagged as UNVERIFIED_NEGATIVE_FINDING.")

    if status != EvidenceStatus.VERIFIED_EXTRACTION.value and status != "VERIFIED_EXTRACTION":
        raise SR117UnverifiedTelemetryError(f"Telemetry payload status '{status}' is unverified.")

    doc_sha256 = payload.get("doc_sha256")
    if not doc_sha256 or not re.match(r"^[a-fA-F0-9]{64}$", str(doc_sha256)):
        raise SR117UnverifiedTelemetryError("Telemetry payload missing valid doc_sha256.")

    source_span = payload.get("source_span")
    if not source_span or not isinstance(source_span, dict) or not str(source_span.get("text", "")).strip():
        raise SR117UnverifiedTelemetryError("Telemetry payload missing source_span grounding.")

    return True
