import time
import json
import pytest
from src.ingestion.zero_trust.sts_auth import STSService, EphemeralToken
from src.ingestion.zero_trust.mtls_gateway import MTLSGateway, TLSHandshakeError
from src.ingestion.zero_trust.prov_o_middleware import ProvOMiddleware, ProvOChecksumError
from src.ingestion.zero_trust.consensus_engine import TriangulatedConsensusEngine, CoreMetrics
from src.ingestion.zero_trust.schema_validator import (
    BoundedMarketMetrics,
    WhalescannerValidator,
    WhalescannerFiling,
    PhysicsValidationError,
    SyntheticMalwareError,
)
from src.ingestion.zero_trust.quarantine_sink import QuarantineSink
from src.ingestion.zero_trust.fortress_state import FortressStateManager, SystemStatus


def test_sts_ephemeral_token_expiration_and_signing():
    sts = STSService()
    token = sts.issue_token(client_id="node_alpha", ttl_seconds=1.0)
    assert sts.validate_token(token) is True

    # Check 15-min TTL cap
    long_token = sts.issue_token(client_id="node_beta", ttl_seconds=3600.0)
    assert long_token.ttl_seconds == 900.0

    # Simulate token expiration
    time.sleep(1.1)
    assert sts.validate_token(token) is False

    # Test tampering
    tampered_token = EphemeralToken(
        token=long_token.token,
        client_id="node_beta",
        created_at=long_token.created_at,
        ttl_seconds=long_token.ttl_seconds,
        signature="invalid_signature"
    )
    assert sts.validate_token(tampered_token) is False


def test_mtls_gateway_ecdsa_and_downgrade_prevention():
    gateway = MTLSGateway()

    # Valid TLS 1.3 ECDSA connection
    valid_conn = {
        "client_cert": "PRESENT",
        "cert_verified": True,
        "tls_version": "TLSv1.3",
        "cipher_name": "TLS_AES_256_GCM_SHA384",
        "cipher_algo": "ECDSA"
    }
    assert gateway.verify_client_connection(valid_conn) is True

    # TLS Downgrade attempt
    downgrade_conn = dict(valid_conn, tls_version="TLSv1.2")
    with pytest.raises(TLSHandshakeError, match="TLS downgrade attempt detected"):
        gateway.verify_client_connection(downgrade_conn)

    # Missing Client Certificate
    missing_cert_conn = dict(valid_conn, cert_verified=False)
    with pytest.raises(TLSHandshakeError, match="Missing or unverified client certificate"):
        gateway.verify_client_connection(missing_cert_conn)

    # Non-ECDSA Cipher Suite
    rsa_conn = dict(valid_conn, cipher_name="DHE-RSA-AES256-SHA", cipher_algo="RSA")
    with pytest.raises(TLSHandshakeError, match="Non-ECDSA cipher suite used"):
        gateway.verify_client_connection(rsa_conn)


def test_prov_o_checksum_middleware():
    middleware = ProvOMiddleware()
    payload = {"sp500": 5800.0, "vix": 15.5}
    valid_hash = middleware.compute_sha256(payload)

    # Valid Hash Match
    assert middleware.process_payload(payload, valid_hash) == valid_hash

    # Checksum Mismatch Error
    with pytest.raises(ProvOChecksumError):
        middleware.process_payload(payload, "0000000000000000000000000000000000000000000000000000000000000000")


def test_triangulated_consensus_spoofing_detection():
    consensus = TriangulatedConsensusEngine()

    feed_alpha = CoreMetrics(sp500=5800.0, yield_10y=4.2, vix=15.0, btc=65000.0, brent=75.0)
    feed_beta = CoreMetrics(sp500=5805.0, yield_10y=4.21, vix=15.1, btc=65100.0, brent=75.1)
    # Spoofed Feed Gamma with manipulated VIX (vix=45.0) and S&P 500 (sp500=4000.0)
    feed_gamma = CoreMetrics(sp500=4000.0, yield_10y=4.20, vix=45.0, btc=65050.0, brent=75.0)

    feeds = {"alpha": feed_alpha, "beta": feed_beta, "gamma": feed_gamma}
    res = consensus.evaluate_consensus(feeds)

    assert res.spoofing_detected is True
    assert "gamma" in res.severed_feeds
    assert res.valid_feed_count == 2
    assert res.consensus_metrics.vix == pytest.approx(15.05, 0.1)


def test_physics_schema_bounding():
    # Negative VIX physics violation
    with pytest.raises(PhysicsValidationError, match="VIX cannot be negative"):
        BoundedMarketMetrics(sp500=5800.0, yield_10y=4.2, vix=-5.0, btc=65000.0, brent=75.0)

    # Extreme daily variance physics violation (>25%)
    with pytest.raises(PhysicsValidationError, match="S&P 500 daily variance exceeds 25%"):
        BoundedMarketMetrics(
            sp500=5800.0,
            yield_10y=4.2,
            vix=15.0,
            btc=65000.0,
            brent=75.0,
            sp500_daily_variance_pct=30.0
        )


def test_sec_cik_whalescanner_synthetic_malware_detection():
    validator = WhalescannerValidator()

    # Valid filing for MSFT (CIK: 0000789019, ~7.43B shares outstanding)
    valid_filing = WhalescannerFiling(
        filing_type="13D",
        target_cik="0000789019",
        target_ticker="MSFT",
        reported_shares=743000000,  # 10%
        implied_ownership_pct=10.0
    )
    assert validator.validate_filing(valid_filing) is True

    # Hallucinated CIK
    unknown_filing = WhalescannerFiling(
        filing_type="13D",
        target_cik="9999999999",
        target_ticker="FAKE",
        reported_shares=1000,
        implied_ownership_pct=1.0
    )
    with pytest.raises(SyntheticMalwareError, match="not found in SEC database"):
        validator.validate_filing(unknown_filing)

    # Share count mathematical misalignment / synthetic injection
    hallucinated_filing = WhalescannerFiling(
        filing_type="13D",
        target_cik="0000789019",
        target_ticker="MSFT",
        reported_shares=743000000,
        implied_ownership_pct=50.0  # False 50% ownership claim
    )
    with pytest.raises(SyntheticMalwareError, match="mathematically violates"):
        validator.validate_filing(hallucinated_filing)


def test_quarantine_sink_and_fortress_lockdown_fallback(tmp_path):
    sink = QuarantineSink(quarantine_dir=str(tmp_path / "quarantine"))
    state_mgr = FortressStateManager()

    assert state_mgr.system_status == SystemStatus.NORMAL
    assert state_mgr.get_current_allocation()["convex_hunt_allocation_pct"] == 25.0

    # Quarantine failed payload
    filepath = sink.quarantine_payload(
        payload={"spoofed_vix": 99.0},
        failure_reason="VIX spoofing attempt detected",
        error_type="AdversarialSpoofingAttempt"
    )
    assert len(sink.get_quarantined_records()) == 1

    # Trigger consensus failure / Lockdown
    alloc = state_mgr.evaluate_state_override(valid_feed_count=1, spoofing_detected=True)
    assert alloc["system_status"] == "LOCKDOWN"
    assert alloc["fortress_allocation_pct"] == 100.0
    assert alloc["convex_hunt_allocation_pct"] == 0.0
    assert alloc["convex_hunt_execution_allowed"] is False
    assert alloc["asset_class"] == "SUB_90_DAY_SOVEREIGN_PAPER"

    # Restored only via meatspace authorization
    state_mgr.authorize_meatspace_override()
    assert state_mgr.system_status == SystemStatus.NORMAL
    assert state_mgr.get_current_allocation()["convex_hunt_allocation_pct"] == 25.0
