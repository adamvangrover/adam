"""
AFOS V3 Kernel API Router: High-Assurance Ingestion Layer.
Endpoints:
- POST /api/v3/sts/token: Issue 15-minute STS credentials
- POST /api/v3/ingest/telemetry: Triangulated macro telemetry ingestion
- POST /api/v3/ingest/whalescanner: Whalescanner & SEC EDGAR cross-referenced alert ingestion
- GET /api/v3/governance/status: System status and quarantine matrix overview
- POST /api/v3/governance/unlock: Human authorization override unlock
"""

import json
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from src.security.v3_ingestion_defense import (
    EphemeralSTSService,
    PROVOChecksumVerifier,
    RTLSEnforcer,
)
from src.ingestion.consensus_engine import FeedDataPoint, TriangulatedTelemetryEngine
from src.ingestion.deterministic_validator import DeterministicDataValidator
from src.governance.quarantine_matrix import global_quarantine_matrix

router = APIRouter(prefix="/api/v3", tags=["AFOS V3 Kernel Ingestion"])
sts_service = EphemeralSTSService()
consensus_engine = TriangulatedTelemetryEngine(required_feeds_count=3)


class STSTokenRequest(BaseModel):
    node_id: str = Field(..., description="Ingestion node identifier")
    node_role: str = Field(..., description="Whalescanner or Macro Telemetry role")
    requested_ttl_seconds: int = Field(900, description="Requested TTL in seconds (Max 900)")


class TelemetryIngestRequest(BaseModel):
    expected_sha256_hash: str = Field(..., description="W3C PROV-O SHA-256 payload checksum")
    feeds_data: List[FeedDataPoint] = Field(..., description="Simultaneous data points from 3 Tier-1 feeds")
    macro_payload: Dict[str, Any] = Field(..., description="Macro telemetry metrics (sp500_1sec_drop_pct, vix, yield_10y, sofr)")


class WhalescannerIngestRequest(BaseModel):
    expected_sha256_hash: str = Field(..., description="W3C PROV-O SHA-256 payload checksum")
    alert_payload: Dict[str, Any] = Field(..., description="Whalescanner takeover / LBO alert payload")


class HumanUnlockRequest(BaseModel):
    authorization_token: str = Field(..., description="Human override authorization token")


@router.post("/sts/token", status_code=200)
async def issue_sts_token(req: STSTokenRequest):
    """
    Issues a short-lived, ECDSA-signed Security Token Service (STS) credential
    with a maximum 15-minute TTL.
    """
    token_resp = sts_service.issue_token(
        node_id=req.node_id,
        node_role=req.node_role,
        ttl_seconds=req.requested_ttl_seconds
    )
    return token_resp


@router.post("/ingest/telemetry", status_code=200)
async def ingest_macro_telemetry(
    req: TelemetryIngestRequest,
    authorization: str = Header(..., description="STS Bearer Token"),
    x_sts_signature: str = Header(..., description="STS ECDSA signature hex"),
    x_tls_version: str = Header("TLSv1.3"),
    x_ssl_client_verify: str = Header("SUCCESS"),
    x_ssl_client_issuer: str = Header("CN=AFOS-V3-ROOT-CA"),
    x_ssl_client_algo: str = Header("ECDSA-p256")
):
    """
    Ingests macro telemetry enforcing mTLS, STS 15-min token verification,
    W3C PROV-O SHA-256 checksums, triangulated multi-vector consensus, and immutable physical bounds.
    """
    # 1. mTLS Enforcement
    if not RTLSEnforcer.verify_mtls(
        tls_version=x_tls_version,
        client_cert_verified=(x_ssl_client_verify == "SUCCESS"),
        client_cert_issuer=x_ssl_client_issuer,
        client_cert_algo=x_ssl_client_algo
    ):
        global_quarantine_matrix.divert_to_quarantine(
            source_node="mTLS-Gateway",
            reason="mTLS handshake failed or TLS downgrade attempt",
            raw_payload=req.model_dump(),
            forensic_tags=["MTLS_FAILURE", "SECURITY_PERIMETER_BREACH"]
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="mTLS verification failed or TLS downgrade attempt."
        )

    # 2. Ephemeral STS Token Check
    token_str = authorization.replace("Bearer ", "").strip()
    sts_payload = sts_service.verify_token(token_str, x_sts_signature)

    # 3. W3C PROV-O SHA-256 Payload Hash Check
    raw_payload_bytes = json.dumps(req.macro_payload, sort_keys=True).encode("utf-8")
    hash_match, calc_hash = PROVOChecksumVerifier.verify_payload_checksum(raw_payload_bytes, req.expected_sha256_hash)

    if not hash_match:
        # Single bit mismatch -> Air-gapped Quarantine & Circuit Breaker
        global_quarantine_matrix.divert_to_quarantine(
            source_node=sts_payload.get("node_id", "unknown"),
            reason=f"W3C PROV-O SHA-256 Bit Mismatch: Expected {req.expected_sha256_hash}, Got {calc_hash}",
            raw_payload=req.macro_payload,
            forensic_tags=["BIT_MISMATCH", "PROV_O_INTEGRITY_VIOLATION"]
        )
        global_quarantine_matrix.trigger_circuit_breaker(
            reason="W3C PROV-O SHA-256 Checksum mismatch on macro telemetry payload."
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="W3C PROV-O SHA-256 checksum mismatch. Payload diverted to quarantine; Hunt & Kill disabled."
        )

    # 4. Multi-Vector Consensus (Triangulated Telemetry)
    metric_name = req.feeds_data[0].metric_name if req.feeds_data else "MACRO_METRIC"
    consensus_res = consensus_engine.compute_consensus(metric_name, req.feeds_data)

    if not consensus_res.consensus_verified or consensus_res.adversarial_presence_flag:
        global_quarantine_matrix.divert_to_quarantine(
            source_node=sts_payload.get("node_id", "unknown"),
            reason=f"Multi-Vector Consensus Failure / Adversarial Outlier: Severed feeds {consensus_res.severed_feeds}",
            raw_payload=req.model_dump(),
            forensic_tags=["CONSENSUS_FAILURE", "ORDERBOOK_SPOOFING_ATTEMPT"]
        )
        # Fallback: Disable Hunt & Kill and revert 100% to Fortress allocation
        system_state = global_quarantine_matrix.trigger_circuit_breaker(
            reason=f"Consensus verification fell below 100%. Severed anomalous feeds: {consensus_res.severed_feeds}"
        )
        return {
            "status": "QUARANTINED",
            "message": "Consensus verification failed (>0.5 std dev outlier detected). Feed severed and Hunt & Kill disabled.",
            "consensus_result": consensus_res.model_dump(),
            "system_state": system_state.model_dump()
        }

    # 5. Deterministic Physical Bounds Validation
    bounds_res = DeterministicDataValidator.validate_macro_payload(req.macro_payload)
    if not bounds_res.is_valid or bounds_res.malware_injection_attempt:
        global_quarantine_matrix.divert_to_quarantine(
            source_node=sts_payload.get("node_id", "unknown"),
            reason=bounds_res.error_message or "Immutable Schema Physical Bounds Violation",
            raw_payload=req.macro_payload,
            forensic_tags=["MALWARE_INJECTION_ATTEMPT", "PHYSICAL_BOUNDS_VIOLATION"]
        )
        system_state = global_quarantine_matrix.trigger_circuit_breaker(
            reason="Malware injection attempt detected via physical boundary violation."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=bounds_res.error_message
        )

    return {
        "status": "ACCEPTED",
        "consensus_value": consensus_res.consensus_value,
        "sample_std_dev": consensus_res.sample_std_dev,
        "system_state": global_quarantine_matrix.get_system_state().model_dump()
    }


@router.post("/ingest/whalescanner", status_code=200)
async def ingest_whalescanner_alert(
    req: WhalescannerIngestRequest,
    authorization: str = Header(..., description="STS Bearer Token"),
    x_sts_signature: str = Header(..., description="STS ECDSA signature hex"),
    x_tls_version: str = Header("TLSv1.3"),
    x_ssl_client_verify: str = Header("SUCCESS")
):
    """
    Ingests Whalescanner takeover/LBO alerts, enforcing STS verification,
    PROV-O checksums, and SEC EDGAR CIK / share count cross-referencing.
    """
    # 1. Verify STS Token
    token_str = authorization.replace("Bearer ", "").strip()
    sts_payload = sts_service.verify_token(token_str, x_sts_signature)

    # 2. PROV-O SHA-256 Hash Check
    raw_payload_bytes = json.dumps(req.alert_payload, sort_keys=True).encode("utf-8")
    hash_match, calc_hash = PROVOChecksumVerifier.verify_payload_checksum(raw_payload_bytes, req.expected_sha256_hash)

    if not hash_match:
        global_quarantine_matrix.divert_to_quarantine(
            source_node=sts_payload.get("node_id", "whalescanner-node"),
            reason="W3C PROV-O SHA-256 Checksum mismatch on Whalescanner alert",
            raw_payload=req.alert_payload,
            forensic_tags=["BIT_MISMATCH", "WHALESCANNER_CORRUPTION"]
        )
        global_quarantine_matrix.trigger_circuit_breaker(
            reason="PROV-O checksum mismatch on Whalescanner alert."
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="W3C PROV-O Checksum mismatch. Alert diverted to quarantine."
        )

    # 3. SEC EDGAR Cross-Referencing
    cross_res = DeterministicDataValidator.cross_reference_whalescanner_alert(req.alert_payload)
    if not cross_res.is_valid or cross_res.synthetic_filing_detected:
        global_quarantine_matrix.divert_to_quarantine(
            source_node=sts_payload.get("node_id", "whalescanner-node"),
            reason=cross_res.error_message or "Synthetic filing / hallucinated company detected",
            raw_payload=req.alert_payload,
            forensic_tags=["SYNTHETIC_FILING_ATTEMPT", "HALLUCINATED_COMPANY_REJECTED"]
        )
        global_quarantine_matrix.trigger_circuit_breaker(
            reason="Synthetic filing with hallucinated company detected."
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=cross_res.error_message
        )

    return {
        "status": "ACCEPTED",
        "verified_alert": cross_res.sanitized_data,
        "system_state": global_quarantine_matrix.get_system_state().model_dump()
    }


@router.get("/governance/status", status_code=200)
async def get_governance_status():
    """Returns current AFOS V3 Kernel governance and circuit breaker status."""
    return global_quarantine_matrix.get_system_state().model_dump()


@router.post("/governance/unlock", status_code=200)
async def human_override_unlock(req: HumanUnlockRequest):
    """Restores Hunt & Kill Orchestrator and 75% Fortress allocation upon human override authorization."""
    try:
        updated_state = global_quarantine_matrix.human_override_unlock(req.authorization_token)
        return {
            "status": "UNLOCKED",
            "message": "Human override authorization accepted. Hunt & Kill re-enabled.",
            "system_state": updated_state.model_dump()
        }
    except PermissionError as err:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(err)
        )
