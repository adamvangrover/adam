import pytest
from core.remediation.extraction_validator import (
    EvidenceBoundExtractionRecord,
    EvidenceSpan,
    EvidenceStatus,
    SR117UnverifiedTelemetryError,
    validate_extraction_record,
    validate_telemetry_payload,
)

VALID_SHA256 = "a" * 64

def test_valid_extraction_record_passes() -> None:
    record = EvidenceBoundExtractionRecord(
        metric_name="ebitda",
        value=450.0,
        units="USD_MILLIONS",
        status=EvidenceStatus.VERIFIED_EXTRACTION,
        doc_id="CREDIT_AGREEMENT_2026_01",
        doc_sha256=VALID_SHA256,
        timestamp="2026-10-17T00:00:00Z",
        confidence=0.99,
        reconciliation_state="RECONCILED",
        source_span=EvidenceSpan(text="Consolidated EBITDA for the period was $450.0 million.", page_number=42),
    )
    assert validate_extraction_record(record) is True

def test_unverified_status_fails() -> None:
    record = EvidenceBoundExtractionRecord(
        metric_name="ebitda",
        value=450.0,
        units="USD_MILLIONS",
        status=EvidenceStatus.UNVERIFIED_NEGATIVE_FINDING,
        doc_id="CREDIT_AGREEMENT_2026_01",
        doc_sha256=VALID_SHA256,
        timestamp="2026-10-17T00:00:00Z",
        confidence=0.99,
        source_span=EvidenceSpan(text="Consolidated EBITDA for the period was $450.0 million.", page_number=42),
    )
    with pytest.raises(SR117UnverifiedTelemetryError):
        validate_extraction_record(record)

def test_missing_source_span_fails() -> None:
    record = EvidenceBoundExtractionRecord(
        metric_name="ebitda",
        value=450.0,
        units="USD_MILLIONS",
        status=EvidenceStatus.VERIFIED_EXTRACTION,
        doc_id="CREDIT_AGREEMENT_2026_01",
        doc_sha256=VALID_SHA256,
        timestamp="2026-10-17T00:00:00Z",
        confidence=0.99,
        source_span=None,
    )
    with pytest.raises(SR117UnverifiedTelemetryError):
        validate_extraction_record(record)

def test_missing_hash_fails() -> None:
    record = EvidenceBoundExtractionRecord(
        metric_name="ebitda",
        value=450.0,
        units="USD_MILLIONS",
        status=EvidenceStatus.VERIFIED_EXTRACTION,
        doc_id="CREDIT_AGREEMENT_2026_01",
        doc_sha256="invalid_hash",
        timestamp="2026-10-17T00:00:00Z",
        confidence=0.99,
        source_span=EvidenceSpan(text="Consolidated EBITDA for the period was $450.0 million.", page_number=42),
    )
    with pytest.raises(SR117UnverifiedTelemetryError):
        validate_extraction_record(record)

def test_raw_telemetry_payload_validation() -> None:
    valid_payload = {
        "status": "VERIFIED_EXTRACTION",
        "doc_sha256": VALID_SHA256,
        "source_span": {"text": "Verified text span"},
    }
    assert validate_telemetry_payload(valid_payload) is True

    invalid_payload = {
        "status": "VERIFIED_EXTRACTION",
        "doc_sha256": VALID_SHA256,
        "source_span": None,
    }
    with pytest.raises(SR117UnverifiedTelemetryError):
        validate_telemetry_payload(invalid_payload)
