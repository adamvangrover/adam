"""
AFOS V3 Zero-Trust Ingestion Gateway & Cryptographic Provenance Layer
Phase 1 (STS, ECDSA mTLS, W3C PROV-O) & Phase 4 (Air-Gapped Quarantine Sink)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import ssl
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

MAX_STS_TTL_SECONDS: Final[int] = 900  # Strict 15-minute maximum TTL
FORBIDDEN_STATIC_ENV_PATTERNS: Final[tuple[str, ...]] = (
    "API_KEY",
    "SECRET_KEY",
    "STATIC_TOKEN",
    "EDGAR_KEY",
    "MARKET_DATA_TOKEN",
    "PRIVATE_KEY_RAW",
)


class ZeroTrustSecurityError(Exception):
    """Base exception for AFOS V3 Zero-Trust perimeter violations."""


class EphemeralTokenError(ZeroTrustSecurityError):
    """Raised when an STS token is missing, expired, forged, or exceeds TTL policy."""


class MTLSVerificationError(ZeroTrustSecurityError):
    """Raised when a client connection violates ECDSA mTLS or attempts TLS downgrade."""


class ProvOChecksumError(ZeroTrustSecurityError):
    """Raised when an incoming payload fails W3C PROV-O SHA-256 verification."""


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """Immutable forensic record serialized to the air-gapped quarantine sink."""

    quarantine_id: str
    timestamp_utc: str
    failure_category: str
    source_node: str
    reason: str
    payload_sha256: str
    raw_payload_excerpt: str
    prov_o_trace: dict[str, Any] = field(default_factory=dict)


class QuarantineSink:
    """
    Phase 4: Air-Gapped Forensic Logging Sink (/quarantine).
    Serializes failed payloads outside the main execution event loop.
    """

    def __init__(self, sink_path: Path | str = Path("/tmp/afos_v3_quarantine/quarantine.jsonl")) -> None:
        self.sink_path = Path(sink_path)
        self.sink_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_buffer: list[QuarantineRecord] = []
        self._lock = asyncio.Lock()

    async def dump(
        self,
        *,
        failure_category: str,
        source_node: str,
        reason: str,
        raw_payload: bytes | str | dict[str, Any],
        prov_o_trace: dict[str, Any] | None = None,
    ) -> QuarantineRecord:
        if isinstance(raw_payload, dict):
            serialized = json.dumps(raw_payload, sort_keys=True).encode("utf-8")
        elif isinstance(raw_payload, str):
            serialized = raw_payload.encode("utf-8")
        else:
            serialized = raw_payload

        digest = hashlib.sha256(serialized).hexdigest()
        record = QuarantineRecord(
            quarantine_id=f"q-{uuid.uuid4()}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            failure_category=failure_category,
            source_node=source_node,
            reason=reason,
            payload_sha256=digest,
            raw_payload_excerpt=serialized.decode("utf-8", errors="replace")[:4096],
            prov_o_trace=prov_o_trace or {},
        )

        async with self._lock:
            self._memory_buffer.append(record)
            line = json.dumps(asdict(record), sort_keys=True) + "\n"
            await asyncio.to_thread(self._append_to_disk, line)

        return record

    def _append_to_disk(self, line: str) -> None:
        with self.sink_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    @property
    def records(self) -> tuple[QuarantineRecord, ...]:
        return tuple(self._memory_buffer)


def strip_static_credentials(env_mapping: dict[str, str] | None = None) -> list[str]:
    """
    Phase 1.1: Scans runtime environment for hardcoded/static market data or
    SEC EDGAR keys and strips them immediately.
    """
    target_env = env_mapping if env_mapping is not None else os.environ
    stripped_keys: list[str] = []

    for key in list(target_env.keys()):
        upper_key = key.upper()
        if any(pattern in upper_key for pattern in FORBIDDEN_STATIC_ENV_PATTERNS):
            target_env.pop(key, None)
            stripped_keys.append(key)

    return stripped_keys


@dataclass(frozen=True, slots=True)
class EphemeralSTSToken:
    """Short-lived cryptographic token issued by the AFOS V3 STS authority."""

    token_id: str
    node_id: str
    issued_at: float
    expires_at: float
    scope: str
    signature: str


class EphemeralSTSProvider:
    """
    Phase 1.2: Ephemeral Security Token Service (STS) / Vault Authority.
    Enforces a strict maximum 15-minute (900s) TTL on all ingestion node tokens.
    """

    def __init__(self, master_signing_key: bytes | None = None) -> None:
        self._signing_key: bytes = master_signing_key or secrets.token_bytes(32)
        self._revoked_token_ids: set[str] = set()

    def issue_token(
        self,
        node_id: str,
        scope: str = "ingest:metrics",
        ttl_seconds: int = MAX_STS_TTL_SECONDS,
        now_ts: float | None = None,
    ) -> EphemeralSTSToken:
        if ttl_seconds <= 0 or ttl_seconds > MAX_STS_TTL_SECONDS:
            raise EphemeralTokenError(
                f"Requested TTL ({ttl_seconds}s) violates Zero-Trust policy (0 < TTL <= {MAX_STS_TTL_SECONDS}s)."
            )

        issued_at = now_ts if now_ts is not None else time.time()
        expires_at = issued_at + ttl_seconds
        token_id = f"sts-{uuid.uuid4()}"
        canonical_claims = f"{token_id}:{node_id}:{issued_at:.6f}:{expires_at:.6f}:{scope}"
        signature = hmac.new(
            self._signing_key,
            canonical_claims.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return EphemeralSTSToken(
            token_id=token_id,
            node_id=node_id,
            issued_at=issued_at,
            expires_at=expires_at,
            scope=scope,
            signature=signature,
        )

    def verify_token(
        self,
        token: EphemeralSTSToken,
        expected_node_id: str,
        required_scope: str,
        now_ts: float | None = None,
    ) -> bool:
        current_time = now_ts if now_ts is not None else time.time()

        if token.token_id in self._revoked_token_ids:
            raise EphemeralTokenError(f"Token {token.token_id} has been explicitly revoked.")

        if token.node_id != expected_node_id:
            raise EphemeralTokenError(
                f"Token node binding mismatch: expected '{expected_node_id}', got '{token.node_id}'."
            )

        if token.scope != required_scope:
            raise EphemeralTokenError(
                f"Token scope mismatch: expected '{required_scope}', got '{token.scope}'."
            )

        if (token.expires_at - token.issued_at) > MAX_STS_TTL_SECONDS:
            raise EphemeralTokenError("Token lifetime exceeds 15-minute maximum policy.")

        if current_time >= token.expires_at or current_time < token.issued_at:
            raise EphemeralTokenError(f"Token {token.token_id} is expired or not yet valid.")

        canonical_claims = (
            f"{token.token_id}:{token.node_id}:{token.issued_at:.6f}:{token.expires_at:.6f}:{token.scope}"
        )
        expected_sig = hmac.new(
            self._signing_key,
            canonical_claims.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(token.signature, expected_sig):
            raise EphemeralTokenError("Cryptographic signature verification failed for STS token.")

        return True

    def revoke_token(self, token_id: str) -> None:
        self._revoked_token_ids.add(token_id)


def create_ecdsa_mtls_context(
    server_cert_path: Path | None = None,
    server_key_path: Path | None = None,
    ca_bundle_path: Path | None = None,
) -> ssl.SSLContext:
    """
    Phase 1.3: Configures inbound gateway SSLContext for strict Mutual TLS (mTLS)
    with TLS 1.3 / 1.2 minimum and ECDSA authenticated cipher suites.
    Drops all connections attempting TLS downgrades or missing client certificates.
    """
    ctx = ssl.SSLContext(ssl.Purpose.CLIENT_AUTH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.options |= (
        ssl.OP_NO_SSLv2
        | ssl.OP_NO_SSLv3
        | ssl.OP_NO_TLSv1
        | ssl.OP_NO_TLSv1_1
        | ssl.OP_NO_COMPRESSION
        | ssl.OP_SINGLE_ECDH_USE
    )

    # Restrict TLS 1.2 fallback ciphers (if ever negotiated by platform) strictly to ECDHE-ECDSA
    ctx.set_ciphers("ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305")

    if server_cert_path and server_key_path:
        ctx.load_cert_chain(certfile=str(server_cert_path), keyfile=str(server_key_path))
    if ca_bundle_path:
        ctx.load_verify_locations(cafile=str(ca_bundle_path))

    return ctx


@dataclass(frozen=True, slots=True)
class MTLSConnectionHandshake:
    """Represents peer transport metadata inspected by the inbound mTLS enforcement gate."""

    tls_version: str
    cipher_suite: str
    client_cert_present: bool
    client_cert_key_algorithm: str
    client_cert_subject_cn: str


def verify_mtls_handshake(handshake: MTLSConnectionHandshake) -> None:
    """
    Deterministic runtime enforcement for mTLS handshakes.
    Rejects missing client certificates, non-ECDSA keys, or TLS downgrade attempts.
    """
    if handshake.tls_version not in ("TLSv1.3", "TLSv1.2"):
        raise MTLSVerificationError(
            f"TLS downgrade attempt blocked: negotiated protocol '{handshake.tls_version}'."
        )

    if not handshake.client_cert_present or not handshake.client_cert_subject_cn:
        raise MTLSVerificationError("Inbound connection rejected: missing verified mTLS client certificate.")

    if handshake.client_cert_key_algorithm.upper() != "ECDSA":
        raise MTLSVerificationError(
            f"Inbound connection rejected: client certificate uses '{handshake.client_cert_key_algorithm}', "
            "strict ECDSA required."
        )

    if "ECDSA" not in handshake.cipher_suite.upper() and handshake.tls_version != "TLSv1.3":
        raise MTLSVerificationError(
            f"Inbound connection rejected: non-ECDSA cipher suite '{handshake.cipher_suite}'."
        )


@dataclass(frozen=True, slots=True)
class VerifiedEnvelope:
    """Output of the W3C PROV-O & Zero-Trust API Gateway middleware."""

    node_id: str
    payload: dict[str, Any]
    sha256_checksum: str
    prov_o_graph: dict[str, Any]


class ZeroTrustGatewayMiddleware:
    """
    Phase 1.4: W3C PROV-O Hashing & Zero-Trust Inbound Gateway Middleware.
    Verifies mTLS, ephemeral STS tokens, and canonical SHA-256 payload checksums.
    Any mismatch raises a security exception and routes the payload to QuarantineSink.
    """

    def __init__(
        self,
        sts_provider: EphemeralSTSProvider,
        quarantine_sink: QuarantineSink,
    ) -> None:
        self._sts = sts_provider
        self._quarantine = quarantine_sink

    @staticmethod
    def compute_canonical_sha256(raw_payload: bytes | str | dict[str, Any]) -> str:
        if isinstance(raw_payload, dict):
            canonical_bytes = json.dumps(raw_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif isinstance(raw_payload, str):
            parsed = json.loads(raw_payload)
            canonical_bytes = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        else:
            parsed = json.loads(raw_payload.decode("utf-8"))
            canonical_bytes = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    async def ingest_request(
        self,
        *,
        node_id: str,
        raw_payload: bytes | str | dict[str, Any],
        expected_sha256: str,
        sts_token: EphemeralSTSToken,
        mtls_handshake: MTLSConnectionHandshake,
        required_scope: str = "ingest:metrics",
        now_ts: float | None = None,
    ) -> VerifiedEnvelope:
        # 1. Enforce ECDSA mTLS Perimeter
        try:
            verify_mtls_handshake(mtls_handshake)
        except MTLSVerificationError as exc:
            await self._quarantine.dump(
                failure_category="MTLS_PERIMETER_BREACH",
                source_node=node_id,
                reason=str(exc),
                raw_payload=raw_payload,
            )
            raise

        # 2. Enforce Ephemeral STS Token Authentication
        try:
            self._sts.verify_token(
                token=sts_token,
                expected_node_id=node_id,
                required_scope=required_scope,
                now_ts=now_ts,
            )
        except EphemeralTokenError as exc:
            await self._quarantine.dump(
                failure_category="STS_AUTH_FAILURE",
                source_node=node_id,
                reason=str(exc),
                raw_payload=raw_payload,
            )
            raise

        # 3. Parse JSON & Verify W3C PROV-O SHA-256 Checksum
        try:
            if isinstance(raw_payload, dict):
                parsed_payload = raw_payload
            elif isinstance(raw_payload, bytes):
                parsed_payload = json.loads(raw_payload.decode("utf-8"))
            else:
                parsed_payload = json.loads(raw_payload)

            computed_hash = self.compute_canonical_sha256(parsed_payload)
        except Exception as exc:
            await self._quarantine.dump(
                failure_category="MALFORMED_JSON_PAYLOAD",
                source_node=node_id,
                reason=f"Payload deserialization error: {exc}",
                raw_payload=raw_payload,
            )
            raise ProvOChecksumError(f"Malformed JSON payload from {node_id}") from exc

        if not hmac.compare_digest(computed_hash.lower(), expected_sha256.lower()):
            prov_failure_trace = {
                "@context": "http://www.w3.org/ns/prov#",
                "prov:Activity": "afos:ZeroTrustChecksumVerification",
                "prov:wasAttributedTo": node_id,
                "afos:expectedHash": expected_sha256,
                "afos:computedHash": computed_hash,
                "afos:status": "REJECTED_CHECKSUM_MISMATCH",
            }
            await self._quarantine.dump(
                failure_category="PROV_O_CHECKSUM_MISMATCH",
                source_node=node_id,
                reason=f"SHA-256 mismatch: expected {expected_sha256}, computed {computed_hash}",
                raw_payload=parsed_payload,
                prov_o_trace=prov_failure_trace,
            )
            raise ProvOChecksumError(
                f"W3C PROV-O SHA-256 verification failed for node '{node_id}': "
                f"expected {expected_sha256}, computed {computed_hash}."
            )

        prov_o_graph = {
            "@context": "http://www.w3.org/ns/prov#",
            "prov:Entity": {
                "@id": f"urn:sha256:{computed_hash}",
                "prov:generatedAtTime": datetime.now(timezone.utc).isoformat(),
                "afos:sha256": computed_hash,
            },
            "prov:Activity": {
                "@id": f"urn:afos:ingest:{uuid.uuid4()}",
                "prov:used": mtls_handshake.client_cert_subject_cn,
            },
            "prov:Agent": {
                "@id": f"urn:afos:node:{node_id}",
                "afos:stsTokenId": sts_token.token_id,
            },
        }

        return VerifiedEnvelope(
            node_id=node_id,
            payload=parsed_payload,
            sha256_checksum=computed_hash,
            prov_o_graph=prov_o_graph,
        )
