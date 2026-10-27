"""
AFOS V3 Adversarial Unit Test Suite
Verifies Phase 1-4 Zero-Trust Remediation:
- Static credential stripping & 15-minute STS token TTL enforcement
- ECDSA mTLS downgrade rejection & W3C PROV-O SHA-256 tamper detection
- Spoofed VIX outlier rejection (> 0.5 sigma) & negative VIX kernel rejection
- Hallucinated 13D SEC CIK / share-count overflow synthetic malware trapping
- Circuit Breaker LOCKDOWN trigger & 75% Fortress / 25% Convex Hunt preservation
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any

import pytest

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
    return ZeroTrustGatewayMiddleware(sts_provider=sts_provider, quarantine_sink=quarantine_sink)


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
    return DeterministicValidationEngine(quarantine_sink=quarantine_sink, sec_cik_registry=registry)


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


def test_phase1_strip_static_keys_and_enforce_sts_ttl(sts_provider: EphemeralSTSProvider) -> None:
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
        sts_provider.issue_token(node_id="node_alpha", ttl_seconds=901, now_ts=FIXED_NOW)

    # Valid 15-minute token expires after 900 seconds
    token = sts_provider.issue_token(node_id="node_alpha", ttl_seconds=900, now_ts=FIXED_NOW)
    assert sts_provider.verify_token(token, "node_alpha", "ingest:metrics", now_ts=FIXED_NOW + 899) is True

    with pytest.raises(EphemeralTokenError, match="expired"):
        sts_provider.verify_token(token, "node_alpha", "ingest:metrics", now_ts=FIXED_NOW + 901)


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
    with pytest.raises(ProvOChecksumError, match="W3C PROV-O SHA-256 verification failed"):
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

    result = await consensus_engine.execute_consensus_cycle(fetchers, now_ts=FIXED_NOW)

    assert result.verified_nodes == ("node_alpha", "node_beta")
    assert result.severed_nodes == ("node_gamma",)
    assert result.vix == pytest.approx(18.505)
    assert len(result.spoofing_incidents) == 1
    assert result.spoofing_incidents[0]["event"] == "Adversarial_Spoofing_Attempt"
    assert result.spoofing_incidents[0]["node_id"] == "node_gamma"

    # Verify QuarantineSink logged the spoofing attempt
    assert any(r.failure_category == "Adversarial_Spoofing_Attempt" for r in quarantine_sink.records)

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
    flash_crash_spoof = _base_metrics_dict(sp500_price=4000.0, sp500_prev_close=5820.0)
    with pytest.raises(KernelPhysicsViolationError, match="exceeds immutable physical ceiling"):
        await validator.validate_core_metrics("node_beta", flash_crash_spoof)

    assert len(quarantine_sink.records) == 2
    assert all(r.failure_category == "KERNEL_PHYSICS_VIOLATION" for r in quarantine_sink.records)


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
    verified = await validator.validate_whalescanner_filing("edgar_node_1", valid_13d)
    assert verified.target_ticker == "AAPL"

    # 2. Hallucinated 13D with fabricated CIK not in SEC database
    phantom_cik_13d = {
        **valid_13d,
        "target_cik": "0009999999",
        "target_ticker": "FAKE",
    }
    with pytest.raises(SyntheticMalwareFilingError, match="does not exist in verified SEC EDGAR CIK registry"):
        await validator.validate_whalescanner_filing("edgar_node_1", phantom_cik_13d)

    # 3. Hallucinated 13D claiming shares misaligned with outstanding share count
    misaligned_shares_13d = {
        **valid_13d,
        "reported_shares_held": 760_205_000,  # Actually 5.00% of AAPL
        "reported_ownership_pct": 19.50,  # Hallucinated 19.50% activist stake
    }
    with pytest.raises(SyntheticMalwareFilingError, match="Mathematical share-count misalignment"):
        await validator.validate_whalescanner_filing("edgar_node_1", misaligned_shares_13d)

    # 4. Hallucinated 13D reporting more shares than total shares outstanding
    overflow_shares_13d = {
        **valid_13d,
        "reported_shares_held": 20_000_000_000,
        "reported_ownership_pct": 99.0,
    }
    with pytest.raises(SyntheticMalwareFilingError, match="exceeds total outstanding shares"):
        await validator.validate_whalescanner_filing("edgar_node_1", overflow_shares_13d)

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
    valid_sig = hmac.new(MEATSPACE_SECRET, challenge.encode("utf-8"), "sha256").hexdigest()
    assert state_manager.authorize_meatspace_override(valid_sig, challenge) is True
    assert state_manager.SYSTEM_STATUS == SystemStatus.NOMINAL.value

    restored = state_manager.compute_routing_allocation(total_nav=200_000_000.0)
    assert restored.convex_hunt_execution_enabled is True
    assert restored.convex_hunt_active_nav == 50_000_000.0
