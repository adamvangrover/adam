"""
Adversarial Unit Test Suite for AFOS V3 Zero-Trust Remediation.

Suite A (FastAPI & src.zero_trust Ingestion Layer):
1. Short-lived STS token issuance & max 15-min TTL expiry.
2. mTLS ECDSA enforcement, TLS downgrade rejection, and W3C PROV-O SHA-256 Checksum error.
3. Anti-spoofing triangulated consensus & outlier rejection (>0.5 std dev).
4. Rigid Pydantic schema physical bounds (VIX < 0 rejection, variance > 25% rejection).
5. SEC CIK & share alignment cross-referencing for Whalescanner (synthetic malware filings).
6. Forensic quarantine logging sink.
7. Circuit Breaker state override -> SYSTEM_STATUS = LOCKDOWN (75% Fortress / 0% Convex Hunt).

Suite B (afos_v3 Phase 1-4 Core Kernel & Async Consensus):
- Static credential stripping & 15-minute STS token TTL enforcement
- ECDSA mTLS downgrade rejection & W3C PROV-O SHA-256 tamper detection
- Spoofed VIX outlier rejection (> 0.5 sigma) & negative VIX kernel rejection
- Hallucinated 13D SEC CIK / share-count overflow synthetic malware trapping
- Circuit Breaker LOCKDOWN trigger & 75% Fortress / 25% Convex Hunt preservation
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

# --- Suite A Imports (src.zero_trust & src.adam) ---
from src.adam.core.state_manager import StateManager
from src.zero_trust.consensus import (
    TriangulatedConsensusEngine as SrcTriangulatedConsensusEngine,
)
from src.zero_trust.gateway import app, sts_service
from src.zero_trust.middleware import (
    ProvOChecksumError as SrcProvOChecksumError,
    ZeroTrustMiddleware,
)
from src.zero_trust.quarantine import QuarantineMatrix
from src.zero_trust.schemas import MarketMetricsPayload, SECFilingIngestionPayload
from src.zero_trust.sec_validator import SECIntegrityError, WhalescannerSECValidator
from src.zero_trust.sts import SecurityTokenError, SecurityTokenService

# --- Suite B Imports (afos_v3) ---
from afos_v3.consensus.triangulated_engine import (
    AFOSKernelStateManager,
    ConsensusLockdownError,
    NodeFeedBundle,
    SystemStatus,
    TriangulatedConsensusEngine,
)
from afos_v3.gateway.zero_trust_gateway import (
    EphemeralSTSProvider,
    EphemeralTokenError,
    MTLSConnectionHandshake,
    MTLSVerificationError,
    ProvOChecksumError,
    QuarantineSink,
    ZeroTrustGatewayMiddleware,
    strip_static_credentials,
)
from afos_v3.validation.schema_bounds import (
    DeterministicValidationEngine,
    KernelPhysicsViolationError,
    SECCIKEntityRecord,
    SyntheticMalwareFilingError,
)

# ============================================================================
# Constants & Shared Helpers
# ============================================================================

FIXED_NOW: float = 1_793_120_000.0
MEATSPACE_SECRET: bytes = b"afos-v3-yubikey-hardware-hmac-secret-key"


def _valid_mtls(node_id: str) -> MTLSConnectionHandshake:
    return MTLSConnectionHandshake(
        tls_version="TLSv1.3",
        cipher_suite="TLS_AES_256_GCM_SHA384",
        client_cert_present=True,
        client_cert_key_algorithm="ECDSA",
        client_cert_subject_cn=f"{node_id}.institutional.afos.internal",
    )


def _base_metrics_dict(
    *,
    vix: float = 18.50,
    sp500_price: float = 5850.25,
    sp500_prev_close: float = 5820.00,
) -> dict[str, Any]:
    return {
        "timestamp_epoch": FIXED_NOW,
        "sp500_price": sp500_price,
        "sp500_prev_close": sp500_prev_close,
        "us10y_yield": 4.21,
        "vix": vix,
        "btc_usd": 92450.00,
        "brent_crude": 76.40,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture()
def quarantine_sink(tmp_path: Path) -> QuarantineSink:
    return QuarantineSink(sink_path=tmp_path / "quarantine.jsonl")


@pytest.fixture()
def sts_provider() -> EphemeralSTSProvider:
    return EphemeralSTSProvider(master_signing_key=b"sts-master-secret-key-32-bytes!!")


@pytest.fixture()
def gateway(
    sts_provider: EphemeralSTSProvider,
    quarantine_sink: QuarantineSink,
) -> ZeroTrustGatewayMiddleware:
    return ZeroTrustGatewayMiddleware(
        sts_provider=sts_provider, quarantine_sink=quarantine_sink
    )


@pytest.fixture()
def validator(quarantine_sink: QuarantineSink) -> DeterministicValidationEngine:
    registry = {
        "0000320193": SECCIKEntityRecord(
            cik="0000320193",
            ticker="AAPL",
            legal_name="Apple Inc.",
            shares_outstanding=15_204_100_000,
        ),
        "0000789019": SECCIKEntityRecord(
            cik="0000789019",
            ticker="MSFT",
            legal_name="Microsoft Corporation",
            shares_outstanding=7_430_000_000,
        ),
    }
    return DeterministicValidationEngine(
        quarantine_sink=quarantine_sink, sec_cik_registry=registry
    )


@pytest.fixture()
def state_manager() -> AFOSKernelStateManager:
    return AFOSKernelStateManager(meatspace_auth_secret=MEATSPACE_SECRET)


@pytest.fixture()
def consensus_engine(
    gateway: ZeroTrustGatewayMiddleware,
    validator: DeterministicValidationEngine,
    state_manager: AFOSKernelStateManager,
    quarantine_sink: QuarantineSink,
) -> TriangulatedConsensusEngine:
    return TriangulatedConsensusEngine(
        gateway=gateway,
        validator=validator,
        state_manager=state_manager,
        quarantine_sink=quarantine_sink,
    )


# ============================================================================
# Suite A: FastAPI & src.zero_trust Ingestion Tests
# ============================================================================


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
        "brent_crude": 82.0,
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    correct_checksum = hashlib.sha256(payload_bytes).hexdigest()

    # 1. Missing checksum header -> ProvOChecksumError (422)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-SSL-Protocol": "TLSv1.3",
        "X-SSL-Client-Cert-Alg": "ECDSA",
        "X-SSL-Client-Verify": "SUCCESS",
    }
    res = client.post("/api/v3/ingest/metrics", content=payload_bytes, headers=headers)
    assert res.status_code == 422
    assert "Checksum" in res.json()["detail"]

    # 2. Corrupted checksum -> ProvOChecksumError (422)
    bad_headers = dict(headers)
    bad_headers["X-PROV-O-Checksum"] = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    res = client.post(
        "/api/v3/ingest/metrics", content=payload_bytes, headers=bad_headers
    )
    assert res.status_code == 422
    assert "Mismatch" in res.json()["detail"]

    # 3. TLS Downgrade Attempt -> Rejection (403)
    downgrade_headers = dict(headers)
    downgrade_headers["X-PROV-O-Checksum"] = correct_checksum
    downgrade_headers["X-SSL-Protocol"] = "TLSv1"
    res = client.post(
        "/api/v3/ingest/metrics", content=payload_bytes, headers=downgrade_headers
    )
    assert res.status_code == 403
    assert "downgrade" in res.json()["detail"].lower()

    # 4. Valid mTLS + PROV-O Checksum -> ACCEPTED (200)
    valid_headers = dict(headers)
    valid_headers["X-PROV-O-Checksum"] = correct_checksum
    res = client.post(
        "/api/v3/ingest/metrics", content=payload_bytes, headers=valid_headers
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ACCEPTED"


def test_triangulated_consensus_outlier_rejection():
    consensus = SrcTriangulatedConsensusEngine(deviation_threshold_std=0.5)

    # Inject spoofed VIX feed (15.0, 15.1, 45.0 - heavy outlier)
    vix_feeds = {
        "Node_Bloomberg": 15.0,
        "Node_Refinitiv": 15.1,
        "Node_Spoofed_Attacker": 45.0,
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
            brent_crude=80.0,
        )

    # 2. S&P 500 Daily Variance > 25% violates bounds
    with pytest.raises(ValueError, match="exceeds rigid limit of 25%"):
        MarketMetricsPayload(
            vix=20.0,
            sp500_daily_variance=30.0,
            sp500_index=5000.0,
            treasury_10y_yield=4.0,
            btc_usd=60000.0,
            brent_crude=80.0,
        )


def test_sec_whalescanner_cik_and_share_alignment():
    validator = WhalescannerSECValidator()

    # 1. Valid MSFT 13D filing
    valid_payload = {
        "filing_type": "13D",
        "target_cik": "0000789019",
        "target_ticker": "MSFT",
        "shares_reported": 50000000,
        "total_shares_outstanding": 7430000000,
    }
    res = validator.validate_filing_payload(valid_payload)
    assert res["verified"] is True
    assert res["company_name"] == "MICROSOFT CORP"

    # 2. Hallucinated / Synthetic CIK -> SECIntegrityError
    hallucinated_cik = dict(valid_payload)
    hallucinated_cik["target_cik"] = "9999999999"
    with pytest.raises(
        SECIntegrityError, match="not found in official SEC CIK database"
    ):
        validator.validate_filing_payload(hallucinated_cik)

    # 3. Share alignment physics violation (shares reported > total outstanding) -> SECIntegrityError
    impossible_shares = dict(valid_payload)
    impossible_shares["shares_reported"] = 999999999999
    with pytest.raises(
        SECIntegrityError, match="exceed official outstanding shares"
    ):
        validator.validate_filing_payload(impossible_shares)


def test_circuit_breaker_fortress_fallback_override():
    sm = StateManager()
    assert sm.system_status == "NORMAL"
    assert sm.allocations["convex_hunt"] == 0.25

    # Trigger circuit breaker on spoofing or consensus failure
    override_res = sm.trigger_circuit_breaker(
        reason="Multi-node spoofing cascade detected"
    )
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
        headers={"X-PROV-O-Checksum": "bad"},
    )
    assert entry["quarantine_id"].startswith("Q-")
    assert entry["rejection_reason"] == "Spoofed VIX payload injected"
    assert len(qm.get_quarantined_records()) == 1


# ============================================================================
# Suite B: AFOS V3 Phase 1-4 Core Kernel & Async Consensus Tests
# ============================================================================


def test_phase1_strip_static_keys_and_enforce_sts_ttl(
    sts_provider: EphemeralSTSProvider,
) -> None:
    mock_env = {
        "PATH": "/usr/bin",
        "BLOOMBERG_API_KEY": "legacy-static-key-123",
        "SEC_EDGAR_KEY": "static-edgar-secret",
        "AFOS_ENV": "production",
    }
    stripped = strip_static_credentials(mock_env)
    assert set(stripped) == {"BLOOMBERG_API_KEY", "SEC_EDGAR_KEY"}
    assert "BLOOMBERG_API_KEY" not in mock_env
    assert mock_env["AFOS_ENV"] == "production"

    # Reject TTL > 15 minutes (900 seconds)
    with pytest.raises(EphemeralTokenError, match="violates Zero-Trust policy"):
        sts_provider.issue_token(
            node_id="node_alpha", ttl_seconds=901, now_ts=FIXED_NOW
        )

    # Valid 15-minute token expires after 900 seconds
    token = sts_provider.issue_token(
        node_id="node_alpha", ttl_seconds=900, now_ts=FIXED_NOW
    )
    assert (
        sts_provider.verify_token(
            token, "node_alpha", "ingest:metrics", now_ts=FIXED_NOW + 899
        )
        is True
    )

    with pytest.raises(EphemeralTokenError, match="expired"):
        sts_provider.verify_token(
            token, "node_alpha", "ingest:metrics", now_ts=FIXED_NOW + 901
        )


@pytest.mark.asyncio()
async def test_phase1_mtls_downgrade_and_prov_o_checksum_tampering(
    gateway: ZeroTrustGatewayMiddleware,
    sts_provider: EphemeralSTSProvider,
    quarantine_sink: QuarantineSink,
) -> None:
    payload = _base_metrics_dict()
    valid_hash = gateway.compute_canonical_sha256(payload)
    token = sts_provider.issue_token(node_id="node_alpha", now_ts=FIXED_NOW)

    # 1. Attempt TLS 1.1 downgrade attack
    downgraded_mtls = MTLSConnectionHandshake(
        tls_version="TLSv1.1",
        cipher_suite="ECDHE-ECDSA-AES256-SHA",
        client_cert_present=True,
        client_cert_key_algorithm="ECDSA",
        client_cert_subject_cn="node_alpha",
    )
    with pytest.raises(MTLSVerificationError, match="TLS downgrade attempt blocked"):
        await gateway.ingest_request(
            node_id="node_alpha",
            raw_payload=payload,
            expected_sha256=valid_hash,
            sts_token=token,
            mtls_handshake=downgraded_mtls,
            now_ts=FIXED_NOW,
        )

    # 2. Attempt W3C PROV-O payload tampering in transit (checksum mismatch)
    tampered_payload = {**payload, "vix": 45.0}
    with pytest.raises(
        ProvOChecksumError, match="W3C PROV-O SHA-256 verification failed"
    ):
        await gateway.ingest_request(
            node_id="node_alpha",
            raw_payload=tampered_payload,
            expected_sha256=valid_hash,
            sts_token=token,
            mtls_handshake=_valid_mtls("node_alpha"),
            now_ts=FIXED_NOW,
        )

    assert len(quarantine_sink.records) == 2
    assert quarantine_sink.records[0].failure_category == "MTLS_PERIMETER_BREACH"
    assert quarantine_sink.records[1].failure_category == "PROV_O_CHECKSUM_MISMATCH"


@pytest.mark.asyncio()
async def test_phase2_spoofed_vix_outlier_rejection_and_feed_severing(
    consensus_engine: TriangulatedConsensusEngine,
    gateway: ZeroTrustGatewayMiddleware,
    sts_provider: EphemeralSTSProvider,
    state_manager: AFOSKernelStateManager,
    quarantine_sink: QuarantineSink,
) -> None:
    """
    Injects a spoofed VIX spike on node_gamma (VIX=39.80) while node_alpha (18.50)
    and node_beta (18.51) report genuine market telemetry.
    Verifies > 0.5 sigma outlier detection, node_gamma disconnection,
    Adversarial_Spoofing_Attempt quarantine logging, and uninterrupted 75/25 routing.
    """

    def _make_fetcher(node_id: str, vix_value: float):
        async def _fetch() -> NodeFeedBundle:
            data = _base_metrics_dict(vix=vix_value)
            checksum = gateway.compute_canonical_sha256(data)
            token = sts_provider.issue_token(node_id=node_id, now_ts=FIXED_NOW)
            return NodeFeedBundle(
                node_id=node_id,
                raw_payload=data,
                expected_sha256=checksum,
                sts_token=token,
                mtls_handshake=_valid_mtls(node_id),
            )

        return _fetch

    fetchers = {
        "node_alpha": _make_fetcher("node_alpha", 18.50),
        "node_beta": _make_fetcher("node_beta", 18.51),
        "node_gamma": _make_fetcher("node_gamma", 39.80),  # Adversarial VIX spoof
    }

    result = await consensus_engine.execute_consensus_cycle(
        fetchers, now_ts=FIXED_NOW
    )

    assert result.verified_nodes == ("node_alpha", "node_beta")
    assert result.severed_nodes == ("node_gamma",)
    assert result.vix == pytest.approx(18.505)
    assert len(result.spoofing_incidents) == 1
    assert result.spoofing_incidents[0]["event"] == "Adversarial_Spoofing_Attempt"
    assert result.spoofing_incidents[0]["node_id"] == "node_gamma"

    # Verify QuarantineSink logged the spoofing attempt
    assert any(
        r.failure_category == "Adversarial_Spoofing_Attempt"
        for r in quarantine_sink.records
    )

    # Verify system remains NOMINAL with 75% Fortress / 25% Convex Hunt active
    assert state_manager.SYSTEM_STATUS == "NOMINAL"
    routing = state_manager.compute_routing_allocation(total_nav=100_000_000.0)
    assert routing.fortress_sovereign_paper_nav == 75_000_000.0
    assert routing.convex_hunt_active_nav == 25_000_000.0
    assert routing.convex_hunt_execution_enabled is True


@pytest.mark.asyncio()
async def test_phase3_negative_vix_and_extreme_sp500_variance_kernel_rejection(
    validator: DeterministicValidationEngine,
    quarantine_sink: QuarantineSink,
) -> None:
    # 1. Impossible negative VIX injection
    negative_vix_payload = _base_metrics_dict(vix=-3.25)
    with pytest.raises(KernelPhysicsViolationError, match="vix"):
        await validator.validate_core_metrics("node_alpha", negative_vix_payload)

    # 2. S&P 500 daily variance exceeding 25% physical ceiling
    flash_crash_spoof = _base_metrics_dict(
        sp500_price=4000.0, sp500_prev_close=5820.0
    )
    with pytest.raises(
        KernelPhysicsViolationError, match="exceeds immutable physical ceiling"
    ):
        await validator.validate_core_metrics("node_beta", flash_crash_spoof)

    assert len(quarantine_sink.records) == 2
    assert all(
        r.failure_category == "KERNEL_PHYSICS_VIOLATION"
        for r in quarantine_sink.records
    )


@pytest.mark.asyncio()
async def test_phase3_hallucinated_13d_filings_flagged_as_synthetic_malware(
    validator: DeterministicValidationEngine,
    quarantine_sink: QuarantineSink,
) -> None:
    # 1. Valid 13D filing for AAPL (5.00% ownership = 760,205,000 shares of 15,204,100,000)
    valid_13d = {
        "accession_number": "0001193125-26-884210",
        "form_type": "SC 13D",
        "filer_cik": "0001067983",
        "target_cik": "0000320193",
        "target_ticker": "AAPL",
        "reported_shares_held": 760_205_000,
        "reported_ownership_pct": 5.00,
    }
    verified = await validator.validate_whalescanner_filing(
        "edgar_node_1", valid_13d
    )
    assert verified.target_ticker == "AAPL"

    # 2. Hallucinated 13D with fabricated CIK not in SEC database
    phantom_cik_13d = {
        **valid_13d,
        "target_cik": "0009999999",
        "target_ticker": "FAKE",
    }
    with pytest.raises(
        SyntheticMalwareFilingError,
        match="does not exist in verified SEC EDGAR CIK registry",
    ):
        await validator.validate_whalescanner_filing("edgar_node_1", phantom_cik_13d)

    # 3. Hallucinated 13D claiming shares misaligned with outstanding share count
    misaligned_shares_13d = {
        **valid_13d,
        "reported_shares_held": 760_205_000,  # Actually 5.00% of AAPL
        "reported_ownership_pct": 19.50,  # Hallucinated 19.50% activist stake
    }
    with pytest.raises(
        SyntheticMalwareFilingError, match="Mathematical share-count misalignment"
    ):
        await validator.validate_whalescanner_filing(
            "edgar_node_1", misaligned_shares_13d
        )

    # 4. Hallucinated 13D reporting more shares than total shares outstanding
    overflow_shares_13d = {
        **valid_13d,
        "reported_shares_held": 20_000_000_000,
        "reported_ownership_pct": 99.0,
    }
    with pytest.raises(
        SyntheticMalwareFilingError, match="exceeds total outstanding shares"
    ):
        await validator.validate_whalescanner_filing(
            "edgar_node_1", overflow_shares_13d
        )

    categories = [r.failure_category for r in quarantine_sink.records]
    assert "SYNTHETIC_MALWARE_UNVERIFIED_CIK" in categories
    assert "SYNTHETIC_MALWARE_MATH_MISALIGNMENT" in categories
    assert "SYNTHETIC_MALWARE_SHARE_COUNT_OVERFLOW" in categories


@pytest.mark.asyncio()
async def test_phase4_multi_feed_compromise_triggers_lockdown_and_fortress_fallback(
    consensus_engine: TriangulatedConsensusEngine,
    gateway: ZeroTrustGatewayMiddleware,
    sts_provider: EphemeralSTSProvider,
    state_manager: AFOSKernelStateManager,
) -> None:
    """
    Simulates simultaneous spoofing / physics corruption across 2 of 3 feeds.
    Verifies SYSTEM_STATUS transitions to LOCKDOWN, 75% Fortress allocation is maintained,
    and 25% Convex Hunt execution is halted until meatspace hardware sign-off.
    """

    async def _good_node() -> NodeFeedBundle:
        data = _base_metrics_dict(vix=18.50)
        return NodeFeedBundle(
            node_id="node_alpha",
            raw_payload=data,
            expected_sha256=gateway.compute_canonical_sha256(data),
            sts_token=sts_provider.issue_token("node_alpha", now_ts=FIXED_NOW),
            mtls_handshake=_valid_mtls("node_alpha"),
        )

    async def _corrupt_physics_node() -> NodeFeedBundle:
        data = _base_metrics_dict(vix=-12.0)  # Violates VIX >= 0 bound
        return NodeFeedBundle(
            node_id="node_beta",
            raw_payload=data,
            expected_sha256=gateway.compute_canonical_sha256(data),
            sts_token=sts_provider.issue_token("node_beta", now_ts=FIXED_NOW),
            mtls_handshake=_valid_mtls("node_beta"),
        )

    async def _bad_checksum_node() -> NodeFeedBundle:
        data = _base_metrics_dict(vix=18.50)
        return NodeFeedBundle(
            node_id="node_gamma",
            raw_payload=data,
            expected_sha256="0" * 64,  # Invalid PROV-O SHA-256
            sts_token=sts_provider.issue_token("node_gamma", now_ts=FIXED_NOW),
            mtls_handshake=_valid_mtls("node_gamma"),
        )

    fetchers = {
        "node_alpha": _good_node,
        "node_beta": _corrupt_physics_node,
        "node_gamma": _bad_checksum_node,
    }

    with pytest.raises(ConsensusLockdownError, match="Gateway/Schema quorum failure"):
        await consensus_engine.execute_consensus_cycle(fetchers, now_ts=FIXED_NOW)

    assert state_manager.SYSTEM_STATUS == SystemStatus.LOCKDOWN.value

    # Verify Fortress Fallback Override: 75% Fortress preserved, 25% Convex Hunt halted
    directive = state_manager.compute_routing_allocation(total_nav=200_000_000.0)
    assert directive.system_status is SystemStatus.LOCKDOWN
    assert directive.fortress_sovereign_paper_nav == 150_000_000.0
    assert directive.convex_hunt_active_nav == 0.0
    assert directive.convex_hunt_standby_cash_nav == 50_000_000.0
    assert directive.convex_hunt_execution_enabled is False

    # Verify manual meatspace hardware authorization restores NOMINAL routing
    challenge = "restore-afos-v3-2026-10-27T21:23:00Z"
    valid_sig = hmac.new(
        MEATSPACE_SECRET, challenge.encode("utf-8"), "sha256"
    ).hexdigest()
    assert state_manager.authorize_meatspace_override(valid_sig, challenge) is True
    assert state_manager.SYSTEM_STATUS == SystemStatus.NOMINAL.value

    restored = state_manager.compute_routing_allocation(total_nav=200_000_000.0)
    assert restored.convex_hunt_execution_enabled is True
    assert restored.convex_hunt_active_nav == 50_000_000.0