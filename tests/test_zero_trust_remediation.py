"""
Adversarial Unit Test Suite for AFOS V3 Zero-Trust Remediation.
Tests:
1. Short-lived STS token issuance & max 15-min TTL expiry.
2. mTLS ECDSA enforcement, TLS downgrade rejection, and W3C PROV-O SHA-256 Checksum error.
3. Anti-spoofing triangulated consensus & outlier rejection (>0.5 std dev).
4. Rigid Pydantic schema physical bounds (VIX < 0 rejection, variance > 25% rejection).
5. SEC CIK & share alignment cross-referencing for Whalescanner (synthetic malware filings).
6. Forensic quarantine logging sink.
7. Circuit Breaker state override -> SYSTEM_STATUS = LOCKDOWN (75% Fortress / 0% Convex Hunt).
"""

import pytest
import time
import hashlib
import json
from fastapi.testclient import TestClient

from src.zero_trust.sts import SecurityTokenService, SecurityTokenError
from src.zero_trust.middleware import ZeroTrustMiddleware, ProvOChecksumError
from src.zero_trust.consensus import TriangulatedConsensusEngine
from src.zero_trust.schemas import MarketMetricsPayload, SECFilingIngestionPayload
from src.zero_trust.sec_validator import WhalescannerSECValidator, SECIntegrityError
from src.zero_trust.quarantine import QuarantineMatrix
from src.zero_trust.gateway import app, sts_service
from src.adam.core.state_manager import StateManager


@pytest.fixture
def client():
    return TestClient(app)


def test_sts_token_lifecycle():
    sts = SecurityTokenService(max_ttl_seconds=900)

    # 1. Test normal token
    res = sts.issue_token(client_id="node-alpha", ttl_seconds=300)
    assert res["ttl_seconds"] == 300
    claims = sts.validate_token(res["token"])
    assert claims["client_id"] == "node-alpha"

    # 2. Test max TTL rigid cap at 15 minutes (900s)
    res_max = sts.issue_token(client_id="node-beta", ttl_seconds=3600)
    assert res_max["ttl_seconds"] == 900

    # 3. Test expired token
    expired_token = f"node-alpha:{int(time.time()) - 100}:signature123"
    with pytest.raises(SecurityTokenError, match="expired"):
        sts.validate_token(expired_token)


def test_provo_checksum_and_mtls_enforcement(client):
    token_info = sts_service.issue_token("test-node")
    token = token_info["token"]

    payload_dict = {
        "vix": 18.5,
        "sp500_daily_variance": 1.2,
        "sp500_index": 5100.0,
        "treasury_10y_yield": 4.25,
        "btc_usd": 65000.0,
        "brent_crude": 82.0
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    correct_checksum = hashlib.sha256(payload_bytes).hexdigest()

    # 1. Missing checksum header -> ProvOChecksumError (422)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-SSL-Protocol": "TLSv1.3",
        "X-SSL-Client-Cert-Alg": "ECDSA",
        "X-SSL-Client-Verify": "SUCCESS"
    }
    res = client.post("/api/v3/ingest/metrics", content=payload_bytes, headers=headers)
    assert res.status_code == 422
    assert "Checksum" in res.json()["detail"]

    # 2. Corrupted checksum -> ProvOChecksumError (422)
    bad_headers = dict(headers)
    bad_headers["X-PROV-O-Checksum"] = "0000000000000000000000000000000000000000000000000000000000000000"
    res = client.post("/api/v3/ingest/metrics", content=payload_bytes, headers=bad_headers)
    assert res.status_code == 422
    assert "Mismatch" in res.json()["detail"]

    # 3. TLS Downgrade Attempt -> Rejection (403)
    downgrade_headers = dict(headers)
    downgrade_headers["X-PROV-O-Checksum"] = correct_checksum
    downgrade_headers["X-SSL-Protocol"] = "TLSv1"
    res = client.post("/api/v3/ingest/metrics", content=payload_bytes, headers=downgrade_headers)
    assert res.status_code == 403
    assert "downgrade" in res.json()["detail"].lower()

    # 4. Valid mTLS + PROV-O Checksum -> ACCEPTED (200)
    valid_headers = dict(headers)
    valid_headers["X-PROV-O-Checksum"] = correct_checksum
    res = client.post("/api/v3/ingest/metrics", content=payload_bytes, headers=valid_headers)
    assert res.status_code == 200
    assert res.json()["status"] == "ACCEPTED"


def test_triangulated_consensus_outlier_rejection():
    consensus = TriangulatedConsensusEngine(deviation_threshold_std=0.5)

    # Inject spoofed VIX feed (15.0, 15.1, 45.0 - heavy outlier)
    vix_feeds = {
        "Node_Bloomberg": 15.0,
        "Node_Refinitiv": 15.1,
        "Node_Spoofed_Attacker": 45.0
    }

    consensus_val, severed, logs = consensus.evaluate_metric_consensus("VIX", vix_feeds)

    assert "Node_Spoofed_Attacker" in severed
    assert len(logs) == 1
    assert logs[0]["event"] == "Adversarial_Spoofing_Attempt"
    assert logs[0]["action"] == "CONNECTION_SEVERED"
    # Consensus should be near 15.05 (mean of valid feeds)
    assert 15.0 <= consensus_val <= 15.1


def test_rigid_schema_physical_bounds():
    # 1. Negative VIX violates physical bounds
    with pytest.raises(ValueError, match="VIX cannot be negative"):
        MarketMetricsPayload(
            vix=-5.0,
            sp500_daily_variance=1.0,
            sp500_index=5000.0,
            treasury_10y_yield=4.0,
            btc_usd=60000.0,
            brent_crude=80.0
        )

    # 2. S&P 500 Daily Variance > 25% violates bounds
    with pytest.raises(ValueError, match="exceeds rigid limit of 25%"):
        MarketMetricsPayload(
            vix=20.0,
            sp500_daily_variance=30.0,
            sp500_index=5000.0,
            treasury_10y_yield=4.0,
            btc_usd=60000.0,
            brent_crude=80.0
        )


def test_sec_whalescanner_cik_and_share_alignment():
    validator = WhalescannerSECValidator()

    # 1. Valid MSFT 13D filing
    valid_payload = {
        "filing_type": "13D",
        "target_cik": "0000789019",
        "target_ticker": "MSFT",
        "shares_reported": 50000000,
        "total_shares_outstanding": 7430000000
    }
    res = validator.validate_filing_payload(valid_payload)
    assert res["verified"] is True
    assert res["company_name"] == "MICROSOFT CORP"

    # 2. Hallucinated / Synthetic CIK -> SECIntegrityError
    hallucinated_cik = dict(valid_payload)
    hallucinated_cik["target_cik"] = "9999999999"
    with pytest.raises(SECIntegrityError, match="not found in official SEC CIK database"):
        validator.validate_filing_payload(hallucinated_cik)

    # 3. Share alignment physics violation (shares reported > total outstanding) -> SECIntegrityError
    impossible_shares = dict(valid_payload)
    impossible_shares["shares_reported"] = 999999999999
    with pytest.raises(SECIntegrityError, match="exceed official outstanding shares"):
        validator.validate_filing_payload(impossible_shares)


def test_circuit_breaker_fortress_fallback_override():
    sm = StateManager()
    assert sm.system_status == "NORMAL"
    assert sm.allocations["convex_hunt"] == 0.25

    # Trigger circuit breaker on spoofing or consensus failure
    override_res = sm.trigger_circuit_breaker(reason="Multi-node spoofing cascade detected")
    assert override_res["system_status"] == "LOCKDOWN"
    assert sm.allocations["fortress"] == 0.75
    assert sm.allocations["convex_hunt"] == 0.00  # Convex hunt executions halted!

    # Meatspace authorization reset
    with pytest.raises(ValueError, match="Unauthorized"):
        sm.reset_lockdown("INVALID_TOKEN")

    reset_res = sm.reset_lockdown("MEATSPACE_AUTH_APPROVED_V3")
    assert reset_res["system_status"] == "NORMAL"
    assert sm.allocations["convex_hunt"] == 0.25


def test_quarantine_matrix_logging_sink():
    qm = QuarantineMatrix()
    entry = qm.quarantine_payload(
        reason="Spoofed VIX payload injected",
        payload={"vix": -99.0},
        headers={"X-PROV-O-Checksum": "bad"}
    )
    assert entry["quarantine_id"].startswith("Q-")
    assert entry["rejection_reason"] == "Spoofed VIX payload injected"
    assert len(qm.get_quarantined_records()) == 1
