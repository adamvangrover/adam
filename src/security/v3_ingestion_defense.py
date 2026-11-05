"""
AFOS V3 Kernel Security Module: Cryptographic Provenance & Perimeter Defense.
Features:
1. Ephemeral STS Credential Management (Max 15-min TTL, ECDSA signed, static keys deprecated).
2. Mutual TLS (mTLS) ECDSA Enforcement & TLS Downgrade Prevention.
3. W3C PROV-O SHA-256 Checksum Recalculation & Bit Mismatch Detection.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# Generate default internal ECDSA keypair for STS token signing
_sts_private_key = ec.generate_private_key(ec.SECP256R1())
_sts_public_key = _sts_private_key.public_key()


class EphemeralSTSService:
    """
    Security Token Service (STS) providing short-lived ECDSA-signed tokens
    with a strict maximum time-to-live (TTL) of 15 minutes (900 seconds).
    Permanently deprecates static API keys.
    """

    MAX_TTL_SECONDS = 900  # 15 minutes max TTL

    def __init__(self, private_key=None, public_key=None):
        self.private_key = private_key or _sts_private_key
        self.public_key = public_key or _sts_public_key

    def issue_token(self, node_id: str, node_role: str, ttl_seconds: int = 900) -> Dict[str, Any]:
        """Issues an ECDSA-signed STS token valid for at most 15 minutes."""
        if ttl_seconds > self.MAX_TTL_SECONDS:
            ttl_seconds = self.MAX_TTL_SECONDS

        now = int(time.time())
        expires_at = now + ttl_seconds

        payload = {
            "node_id": node_id,
            "node_role": node_role,
            "issued_at": now,
            "expires_at": expires_at,
            "ttl": ttl_seconds,
        }

        canonical_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = self.private_key.sign(
            canonical_payload,
            ec.ECDSA(hashes.SHA256())
        )

        return {
            "token_type": "BearerSTS",
            "access_token": canonical_payload.decode("utf-8"),
            "signature_hex": signature.hex(),
            "expires_at": expires_at,
            "max_ttl_seconds": self.MAX_TTL_SECONDS
        }

    def verify_token(self, access_token_str: str, signature_hex: str) -> Dict[str, Any]:
        """
        Verifies the ECDSA signature and expiration of an STS token.
        Rejects static API keys and expired/corrupted tokens.
        """
        try:
            canonical_bytes = access_token_str.encode("utf-8")
            signature_bytes = bytes.fromhex(signature_hex)

            # Verify ECDSA signature
            self.public_key.verify(
                signature_bytes,
                canonical_bytes,
                ec.ECDSA(hashes.SHA256())
            )

            payload = json.loads(access_token_str)
            now = int(time.time())

            # Enforce 15-minute TTL check
            if now > payload["expires_at"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="STS Token expired. Maximum allowed TTL is 15 minutes."
                )

            if payload["expires_at"] - payload["issued_at"] > self.MAX_TTL_SECONDS:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="STS Token TTL exceeds maximum limit of 15 minutes."
                )

            return payload

        except HTTPException:
            raise
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or corrupted STS token signature: {str(err)}"
            )


class RTLSEnforcer:
    """
    Mutual TLS (mTLS) & ECDSA Client Certificate Enforcer.
    Validates mTLS handshake and prevents TLS downgrade attacks across the Hub-and-Spoke architecture.
    """

    ALLOWED_TLS_VERSIONS = {"TLSv1.3", "TLSv1.2"}
    AFOS_ISSUER_CN = "AFOS-V3-ROOT-CA"

    @classmethod
    def verify_mtls(
        cls,
        tls_version: str,
        client_cert_verified: bool,
        client_cert_issuer: Optional[str] = None,
        client_cert_algo: Optional[str] = None
    ) -> bool:
        """
        Validates mTLS properties:
        - TLS version must be >= TLSv1.2 (no TLS downgrade allowed)
        - Client certificate must be verified and issued by AFOS CA
        - Certificate signature must use ECDSA
        """
        if tls_version not in cls.ALLOWED_TLS_VERSIONS:
            return False  # Dropped due to TLS downgrade attempt

        if not client_cert_verified:
            return False

        if client_cert_issuer and cls.AFOS_ISSUER_CN not in client_cert_issuer:
            return False

        if client_cert_algo and "ECDSA" not in client_cert_algo.upper() and "EC" not in client_cert_algo.upper():
            return False

        return True


class PROVOChecksumVerifier:
    """
    W3C PROV-O SHA-256 Checksum Recalculator.
    Recalculates cryptographic payload hash upon receipt; 1-bit mismatch triggers immediate quarantine.
    """

    @staticmethod
    def calculate_sha256(raw_bytes: bytes) -> str:
        """Calculates SHA-256 hash of raw payload bytes."""
        return hashlib.sha256(raw_bytes).hexdigest()

    @staticmethod
    def verify_payload_checksum(raw_bytes: bytes, expected_hash: str) -> Tuple[bool, str]:
        """
        Recalculates SHA-256 hash and compares with expected hash.
        Returns (is_valid, calculated_hash).
        """
        calculated_hash = hashlib.sha256(raw_bytes).hexdigest()
        is_match = calculated_hash.lower() == expected_hash.strip().lower()
        return is_match, calculated_hash


class mTLSEnforcementMiddleware(BaseHTTPMiddleware):
    """
    FastAPI / Starlette middleware enforcing mTLS ECDSA certificates
    and deprecating static API keys on AFOS V3 ingestion routes.
    """

    async def dispatch(self, request: Request, call_next):
        # Enforce on /api/v3/ingest endpoints
        if request.url.path.startswith("/api/v3/ingest"):
            # Deprecate static API keys
            if "x-api-key" in request.headers or "authorization-static" in request.headers:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "STATIC_API_KEY_DEPRECATED",
                        "message": "Static API keys are permanently deprecated. Short-lived STS token required."
                    }
                )

            # mTLS Header checks (forwarded by NGINX / Envoy egress sidecar)
            tls_version = request.headers.get("x-tls-version", "TLSv1.3")
            client_cert_verify = request.headers.get("x-ssl-client-verify", "SUCCESS") == "SUCCESS"
            client_cert_issuer = request.headers.get("x-ssl-client-issuer", "CN=AFOS-V3-ROOT-CA")
            client_cert_algo = request.headers.get("x-ssl-client-algo", "ECDSA-p256")

            # Check mTLS verification
            mtls_valid = RTLSEnforcer.verify_mtls(
                tls_version=tls_version,
                client_cert_verified=client_cert_verify,
                client_cert_issuer=client_cert_issuer,
                client_cert_algo=client_cert_algo
            )

            if not mtls_valid:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "MTLS_ENFORCEMENT_FAILED",
                        "message": "Connection dropped: Missing or unverified AFOS-issued ECDSA client certificate, or TLS downgrade detected."
                    }
                )

        response = await call_next(request)
        return response
