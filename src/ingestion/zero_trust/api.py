from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel
from src.ingestion.zero_trust.sts_auth import STSService, EphemeralToken
from src.ingestion.zero_trust.mtls_gateway import MTLSGateway, TLSHandshakeError
from src.ingestion.zero_trust.prov_o_middleware import ProvOMiddleware, ProvOChecksumError
from src.ingestion.zero_trust.consensus_engine import TriangulatedConsensusEngine, CoreMetrics
from src.ingestion.zero_trust.schema_validator import BoundedMarketMetrics, WhalescannerValidator, WhalescannerFiling, PhysicsValidationError, SyntheticMalwareError
from src.ingestion.zero_trust.quarantine_sink import QuarantineSink
from src.ingestion.zero_trust.fortress_state import FortressStateManager, SystemStatus

router = APIRouter(prefix="/api/v3/zero-trust", tags=["zero-trust-ingestion"])

sts_service = STSService()
mtls_gateway = MTLSGateway()
prov_o_middleware = ProvOMiddleware()
consensus_engine = TriangulatedConsensusEngine()
whalescanner_validator = WhalescannerValidator()
quarantine_sink = QuarantineSink()
fortress_state_manager = FortressStateManager()

class IngestMetricsRequest(BaseModel):
    token: EphemeralToken
    connection_info: Dict[str, Any]
    expected_checksum: str
    feed_data: Dict[str, CoreMetrics]

@router.post("/ingest-metrics")
async def ingest_core_metrics(request: IngestMetricsRequest):
    """
    Zero-Trust Core Metrics Ingestion Endpoint.
    Enforces mTLS, Ephemeral STS Token, PROV-O Checksum, Triangulated Consensus, Physical Bounding,
    Quarantine on failure, and Fortress LOCKDOWN state override.
    """
    # 1. mTLS verification
    try:
        mtls_gateway.verify_client_connection(request.connection_info)
    except TLSHandshakeError as e:
        quarantine_sink.quarantine_payload(request.model_dump(), str(e), "TLSHandshakeError")
        raise HTTPException(status_code=403, detail=str(e))

    # 2. STS Ephemeral Token verification
    if not sts_service.validate_token(request.token):
        err_msg = "STS Token validation failed or expired (max 15-min TTL)"
        quarantine_sink.quarantine_payload(request.model_dump(), err_msg, "STSTokenError")
        raise HTTPException(status_code=401, detail=err_msg)

    # 3. W3C PROV-O Checksum verification
    try:
        raw_feed_dict = {k: v.model_dump() for k, v in request.feed_data.items()}
        prov_o_middleware.process_payload(raw_feed_dict, request.expected_checksum)
    except ProvOChecksumError as e:
        quarantine_sink.quarantine_payload(request.model_dump(), str(e), "ProvOChecksumError")
        raise HTTPException(status_code=422, detail=str(e))

    # 4. Triangulated Consensus (Anti-Spoofing)
    consensus_res = consensus_engine.evaluate_consensus(request.feed_data)

    # Evaluate Fortress Fallback Override
    allocation = fortress_state_manager.evaluate_state_override(
        valid_feed_count=consensus_res.valid_feed_count,
        spoofing_detected=consensus_res.spoofing_detected,
        total_feeds=len(request.feed_data)
    )

    if consensus_res.spoofing_detected:
        quarantine_sink.quarantine_payload(
            request.model_dump(),
            f"Adversarial spoofing detected. Severed feeds: {consensus_res.severed_feeds}",
            "AdversarialSpoofingAttempt"
        )

    # 5. Deterministic Physical Bounds Validation
    try:
        bounded_metrics = BoundedMarketMetrics(
            sp500=consensus_res.consensus_metrics.sp500,
            yield_10y=consensus_res.consensus_metrics.yield_10y,
            vix=consensus_res.consensus_metrics.vix,
            btc=consensus_res.consensus_metrics.btc,
            brent=consensus_res.consensus_metrics.brent
        )
    except PhysicsValidationError as e:
        quarantine_sink.quarantine_payload(request.model_dump(), str(e), "PhysicsValidationError")
        fortress_state_manager.trigger_lockdown(f"Physics validation failure: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "SUCCESS" if fortress_state_manager.system_status == SystemStatus.NORMAL else "LOCKDOWN",
        "consensus": consensus_res.model_dump(),
        "allocation": allocation
    }

@router.get("/quarantine")
async def get_quarantined_payloads():
    """Air-gapped logging sink endpoint for forensic analysis."""
    return {"quarantined_records": quarantine_sink.get_quarantined_records()}
