"""
Zero-Trust API Ingestion Gateway.
Ties together mTLS, STS, validation middleware, triangulated consensus,
SEC CIK cross-referencing, and forensic quarantine routing.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from .sts import SecurityTokenService, SecurityTokenError
from .middleware import ZeroTrustMiddleware, ProvOChecksumError
from .consensus import TriangulatedConsensusEngine
from .sec_validator import WhalescannerSECValidator, SECIntegrityError
from .schemas import MarketMetricsPayload
from .quarantine import QuarantineMatrix
from src.adam.core.state_manager import StateManager

app = FastAPI(title="AFOS V3 Zero-Trust Ingestion Gateway", version="3.0.0")

sts_service = SecurityTokenService()
quarantine_matrix = QuarantineMatrix()
consensus_engine = TriangulatedConsensusEngine()
sec_validator = WhalescannerSECValidator()
state_manager = StateManager()

app.state.sts_service = sts_service
app.state.state_manager = state_manager


@app.post("/api/v3/token/issue")
async def issue_sts_token(client_id: str, ttl_seconds: Optional[int] = 900):
    """Issues short-lived STS token with 15-minute max TTL."""
    return sts_service.issue_token(client_id=client_id, ttl_seconds=ttl_seconds)


@app.post("/api/v3/ingest/metrics")
async def ingest_market_metrics(request: Request):
    """
    Ingests market metrics through Zero-Trust pipeline:
    1. mTLS & STS Token verification
    2. W3C PROV-O SHA-256 Checksum check
    3. Triangulated consensus & outlier rejection
    4. Rigid schema physical bounds check
    On failure: dumps payload to quarantine sink without main loop execution.
    """
    body_bytes = await request.body()
    middleware = ZeroTrustMiddleware(sts_service=sts_service)

    try:
        verified_data = await middleware.verify_request(request, body_bytes)
        raw_payload = verified_data["payload"]

        # Run Pydantic schema physical bounds check
        metrics_payload = MarketMetricsPayload(**raw_payload)

        # If payload contains multi-feed data, process directly; otherwise cross-fetch Tier-1 feeds
        multi_feeds = raw_payload.get("feeds")
        if multi_feeds and isinstance(multi_feeds, dict):
            consensus_res = await consensus_engine.process_metrics_pipeline(multi_feeds)
        else:
            # Cross-reference submitted payload node against Tier-1 institutional feeds
            tier1_feeds = await consensus_engine.fetch_tier1_feeds_concurrently("VIX")
            tier1_feeds["Ingested_Payload_Node"] = metrics_payload.vix

            consensus_res = await consensus_engine.process_metrics_pipeline({
                "VIX": tier1_feeds
            })

        if consensus_res["status"] == "LOCKDOWN":
            # 🛡️ Trigger Circuit Breaker state override -> LOCKDOWN (75% Fortress / 0% Convex Hunt)
            state_manager.trigger_circuit_breaker(reason="Consensus threshold compromised. Multi-node spoofing detected.")
            quarantine_matrix.quarantine_payload(
                reason="Consensus failed threshold -> LOCKDOWN",
                payload=raw_payload,
                headers=dict(request.headers)
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "LOCKDOWN",
                    "allocations": state_manager.allocations,
                    "detail": "Consensus threshold compromised. System in LOCKDOWN mode."
                }
            )

        return {
            "status": "ACCEPTED",
            "provenance_checksum": verified_data["checksum"],
            "consensus": consensus_res["consensus_metrics"]
        }

    except Exception as e:
        # Catch any failure and dump to quarantine
        quarantine_matrix.quarantine_payload(
            reason=str(e),
            payload=body_bytes.decode("utf-8", errors="ignore"),
            headers=dict(request.headers)
        )
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Zero-Trust Ingestion Rejected: {str(e)}"
        )


@app.post("/api/v3/ingest/sec-filing")
async def ingest_sec_filing(request: Request):
    """
    Ingests SEC Whalescanner filing (13D/13F/14A) through Zero-Trust pipeline:
    Cross-references CIK database and outstanding share count alignment.
    Dumps synthetic malware filings into Quarantine Matrix.
    """
    body_bytes = await request.body()
    middleware = ZeroTrustMiddleware(sts_service=sts_service)

    try:
        verified_data = await middleware.verify_request(request, body_bytes)
        raw_payload = verified_data["payload"]

        # Validate against SEC CIK database & share alignment
        validation_res = sec_validator.validate_filing_payload(raw_payload)

        return {
            "status": "ACCEPTED",
            "filing_analysis": validation_res
        }

    except Exception as e:
        quarantine_matrix.quarantine_payload(
            reason=f"SEC Filing Rejected: {str(e)}",
            payload=body_bytes.decode("utf-8", errors="ignore"),
            headers=dict(request.headers)
        )
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Synthetic Filing Rejected: {str(e)}"
        )


@app.get("/quarantine")
async def get_quarantine_logs():
    """Air-gapped quarantine logging sink endpoint for forensic analysis."""
    return {"quarantined_payloads": quarantine_matrix.get_quarantined_records()}
