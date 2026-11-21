"""
Comprehensive Test Suite for AFOS V3 Kernel API Ingestion Layer.
Covers:
1. Ephemeral STS Token Generation, Verification, Expiration (>15 min), and Static Key Deprecation.
2. mTLS ECDSA Enforcement & TLS Downgrade Prevention.
3. W3C PROV-O SHA-256 Checksum Verification & Bit Mismatch Detection.
4. Triangulated Multi-Vector Consensus & 0.5 Std Dev Outlier Rejection.
5. Deterministic Data Validation & Physical Boundary Enforcement (negative VIX, >45% 1-sec drop).
6. SEC EDGAR CIK & Outstanding Share Cross-Referencing for Whalescanner.
7. Air-Gapped Quarantine Sandbox & Hunt & Kill Disabling / Fortress Fallback Circuit Breaker.
"""

import json
import hashlib
import time
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

from src.security.v3_ingestion_defense import (
    EphemeralSTSService,
    PROVOChecksumVerifier,
    RTLSEnforcer,
    mTLSEnforcementMiddleware,
)
from src.ingestion.consensus_engine import FeedDataPoint, TriangulatedTelemetryEngine
from src.ingestion.deterministic_validator import DeterministicDataValidator
from src.governance.quarantine_matrix import QuarantineMatrix, global_quarantine_matrix
from src.api.routers.v3_ingest import router as v3_router

# Setup test FastAPI app
app = FastAPI()
app.add_middleware(mTLSEnforcementMiddleware)
app.include_router(v3_router)

client = TestClient(app)


# --- Phase 1: Cryptographic Provenance & STS Tests ---

def test_sts_token_issuance_and_verification():
    sts = EphemeralSTSService()
    token_resp = sts.issue_token("node-whale-01", "whalescanner", ttl_seconds=600)

    assert token_resp["token_type"] == "BearerSTS"
    assert token_resp["expires_at"] > time.time()
    assert token_resp["max_ttl_seconds"] == 900

    payload = sts.verify_token(token_resp["access_token"], token_resp["signature_hex"])
    assert payload["node_id"] == "node-whale-01"
    assert payload["node_role"] == "whalescanner"


def test_sts_token_ttl_cap_and_expiration():
    sts = EphemeralSTSService()
    # Attempt issuing token exceeding 15 min (e.g. 3600s) -> automatically capped at 900s
    token_resp = sts.issue_token("node-macro-01", "macro_telemetry", ttl_seconds=3600)
    assert token_resp["expires_at"] - token_resp["max_ttl_seconds"] <= int(time.time()) + 1

    # Verified token created in past (expired)
    past_token = sts.issue_token("node-expired", "macro_telemetry", ttl_seconds=-10)
    with pytest.raises(HTTPException) as exc_info:
        sts.verify_token(past_token["access_token"], past_token["signature_hex"])
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_mtls_enforcement_and_tls_downgrade():
    # Valid mTLS
    assert RTLSEnforcer.verify_mtls(
        tls_version="TLSv1.3",
        client_cert_verified=True,
        client_cert_issuer="CN=AFOS-V3-ROOT-CA",
        client_cert_algo="ECDSA-p256"
    ) is True

    # TLS Downgrade attempt (e.g. TLSv1.0 or TLSv1.1)
    assert RTLSEnforcer.verify_mtls(
        tls_version="TLSv1.0",
        client_cert_verified=True,
        client_cert_issuer="CN=AFOS-V3-ROOT-CA",
        client_cert_algo="ECDSA-p256"
    ) is False

    # Non-AFOS Issuer
    assert RTLSEnforcer.verify_mtls(
        tls_version="TLSv1.3",
        client_cert_verified=True,
        client_cert_issuer="CN=UNTRUSTED-CA",
        client_cert_algo="ECDSA-p256"
    ) is False

    # RSA cert instead of ECDSA
    assert RTLSEnforcer.verify_mtls(
        tls_version="TLSv1.3",
        client_cert_verified=True,
        client_cert_issuer="CN=AFOS-V3-ROOT-CA",
        client_cert_algo="RSA-2048"
    ) is False


def test_prov_o_checksum_verification():
    payload_dict = {"sp500_1sec_drop_pct": 2.5, "vix": 18.5}
    payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
    valid_hash = hashlib.sha256(payload_bytes).hexdigest()

    # Valid hash
    match, calc_hash = PROVOChecksumVerifier.verify_payload_checksum(payload_bytes, valid_hash)
    assert match is True
    assert calc_hash == valid_hash

    # Single bit mismatch
    corrupted_hash = valid_hash[:-1] + ("0" if valid_hash[-1] != "0" else "1")
    match_corrupt, _ = PROVOChecksumVerifier.verify_payload_checksum(payload_bytes, corrupted_hash)
    assert match_corrupt is False


# --- Phase 2: Multi-Vector Consensus Tests ---

def test_triangulated_consensus_pass():
    engine = TriangulatedTelemetryEngine(required_feeds_count=3)
    feeds = [
        FeedDataPoint(feed_id="bloomberg-1", feed_name="Bloomberg Tier1", metric_name="SOFR", value=5.32, timestamp=time.time()),
        FeedDataPoint(feed_id="refinitiv-1", feed_name="Refinitiv Tier1", metric_name="SOFR", value=5.32, timestamp=time.time()),
        FeedDataPoint(feed_id="ice-1", feed_name="ICE Data Tier1", metric_name="SOFR", value=5.32, timestamp=time.time()),
    ]

    res = engine.compute_consensus("SOFR", feeds)
    assert res.consensus_verified is True
    assert res.adversarial_presence_flag is False
    assert res.consensus_value == 5.32
    assert len(res.severed_feeds) == 0


def test_triangulated_consensus_outlier_severed():
    engine = TriangulatedTelemetryEngine(required_feeds_count=3)
    # Feed 3 is spoofed / anomalous
    feeds = [
        FeedDataPoint(feed_id="bloomberg-1", feed_name="Bloomberg Tier1", metric_name="VIX", value=18.5, timestamp=time.time()),
        FeedDataPoint(feed_id="refinitiv-1", feed_name="Refinitiv Tier1", metric_name="VIX", value=18.6, timestamp=time.time()),
        FeedDataPoint(feed_id="spoofed-3", feed_name="Adversarial Feed", metric_name="VIX", value=45.0, timestamp=time.time()),
    ]

    res = engine.compute_consensus("VIX", feeds)
    assert res.consensus_verified is False
    assert res.adversarial_presence_flag is True
    assert "spoofed-3" in res.severed_feeds
    assert "bloomberg-1" in res.accepted_feeds
    assert "refinitiv-1" in res.accepted_feeds


# --- Phase 3: Deterministic Data Validation Tests ---

def test_macro_physical_bounds_validation():
    # Valid payload
    valid_payload = {"sp500_1sec_drop_pct": 1.2, "vix": 18.2, "yield_10y": 4.25, "sofr": 5.30}
    res_valid = DeterministicDataValidator.validate_macro_payload(valid_payload)
    assert res_valid.is_valid is True

    # Malware injection attempt: S&P 500 drop 55% in 1 sec
    impossible_drop = {"sp500_1sec_drop_pct": 55.0, "vix": 18.2}
    res_drop = DeterministicDataValidator.validate_macro_payload(impossible_drop)
    assert res_drop.is_valid is False
    assert res_drop.malware_injection_attempt is True
    assert "PHYSICAL BOUND VIOLATION" in res_drop.error_message

    # Negative VIX
    negative_vix = {"sp500_1sec_drop_pct": 0.5, "vix": -5.0}
    res_vix = DeterministicDataValidator.validate_macro_payload(negative_vix)
    assert res_vix.is_valid is False
    assert res_vix.malware_injection_attempt is True


def test_whalescanner_sec_edgar_cross_referencing():
    # Valid alert for Microsoft (CIK: 0000789019)
    valid_alert = {
        "target_cik": "0000789019",
        "target_company_name": "MICROSOFT CORP",
        "reported_outstanding_shares": 7430000000
    }
    res_valid = DeterministicDataValidator.cross_reference_whalescanner_alert(valid_alert)
    assert res_valid.is_valid is True
    assert res_valid.synthetic_filing_detected is False

    # Synthetic filing with hallucinated CIK
    synthetic_alert = {
        "target_cik": "9999999999",
        "target_company_name": "FAKE AI CORP",
        "reported_outstanding_shares": 1000000
    }
    res_synth = DeterministicDataValidator.cross_reference_whalescanner_alert(synthetic_alert)
    assert res_synth.is_valid is False
    assert res_synth.synthetic_filing_detected is True
    assert "SYNTHETIC FILING REJECTED" in res_synth.error_message


# --- Phase 4: Circuit Breaker & Governance Tests ---

def test_quarantine_matrix_and_circuit_breaker():
    qm = QuarantineMatrix()
    assert qm.get_system_state().hunt_and_kill_enabled is True
    assert qm.get_system_state().fortress_allocation_pct == 75.0

    # Divert corrupted payload
    record = qm.divert_to_quarantine("node-01", "SHA256 mismatch", {"data": "bad"}, ["TEST_TAG"])
    assert record.quarantine_id.startswith("QARR-")
    records = qm.list_quarantine_records()
    assert len(records) == 1

    # Trigger circuit breaker fallback
    state = qm.trigger_circuit_breaker("Adversarial spoofing attack detected")
    assert state.hunt_and_kill_enabled is False
    assert state.fortress_allocation_pct == 100.0
    assert state.convex_hunt_allocation_pct == 0.0
    assert state.lockdown_active is True

    # Unlock via human authorization
    unlocked_state = qm.human_override_unlock("HUMAN_OVERRIDE_APPROVED_OFFICER_01")
    assert unlocked_state.hunt_and_kill_enabled is True
    assert unlocked_state.fortress_allocation_pct == 75.0
    assert unlocked_state.lockdown_active is False


# --- Phase 5: API End-to-End Endpoint Integration Tests ---

def test_api_static_key_deprecation():
    # Attempting request with legacy static x-api-key header -> HTTP 401
    resp = client.post("/api/v3/ingest/telemetry", headers={"x-api-key": "legacy-static-key-123"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "STATIC_API_KEY_DEPRECATED"


def test_api_end_to_end_telemetry_flow():
    # 1. Obtain STS Token
    sts_resp = client.post("/api/v3/sts/token", json={"node_id": "node-test-01", "node_role": "telemetry", "requested_ttl_seconds": 600})
    assert sts_resp.status_code == 200
    sts_data = sts_resp.json()

    # Prepare macro payload
    macro_payload = {"sp500_1sec_drop_pct": 0.5, "vix": 18.5, "yield_10y": 4.2, "sofr": 5.3}
    raw_payload_bytes = json.dumps(macro_payload, sort_keys=True).encode("utf-8")
    expected_hash = hashlib.sha256(raw_payload_bytes).hexdigest()

    feeds_data = [
        {"feed_id": "bbg-1", "feed_name": "Bloomberg", "metric_name": "SOFR", "value": 5.3, "timestamp": time.time()},
        {"feed_id": "ref-1", "feed_name": "Refinitiv", "metric_name": "SOFR", "value": 5.3, "timestamp": time.time()},
        {"feed_id": "ice-1", "feed_name": "ICE", "metric_name": "SOFR", "value": 5.3, "timestamp": time.time()},
    ]

    # 2. Ingest Telemetry with STS Token and mTLS headers
    headers = {
        "authorization": f"Bearer {sts_data['access_token']}",
        "x-sts-signature": sts_data["signature_hex"],
        "x-tls-version": "TLSv1.3",
        "x-ssl-client-verify": "SUCCESS",
        "x-ssl-client-issuer": "CN=AFOS-V3-ROOT-CA",
        "x-ssl-client-algo": "ECDSA-p256"
    }

    resp = client.post("/api/v3/ingest/telemetry", json={
        "expected_sha256_hash": expected_hash,
        "feeds_data": feeds_data,
        "macro_payload": macro_payload
    }, headers=headers)

    assert resp.status_code == 200
    assert resp.json()["status"] == "ACCEPTED"
    assert resp.json()["consensus_value"] == 5.3
