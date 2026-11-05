import time
import hashlib
import hmac
import secrets
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class EphemeralToken(BaseModel):
    token: str
    client_id: str
    created_at: float
    ttl_seconds: float = 900.0  # 15 minutes max TTL
    signature: str

class STSService:
    """
    Security Token Service (STS) for short-lived (15-min TTL) zero-trust authentication tokens.
    Replaces static API keys and static environment credentials.
    Uses HMAC-SHA256 signing with ephemeral key material.
    """
    def __init__(self, secret_key: Optional[str] = None):
        self._secret_key = secret_key or secrets.token_hex(32)
        self._revoked_tokens = set()

    def issue_token(self, client_id: str, ttl_seconds: float = 900.0) -> EphemeralToken:
        if ttl_seconds > 900.0:
            ttl_seconds = 900.0  # Strict enforcement of max 15-minute TTL

        created_at = time.time()
        raw_payload = f"{client_id}:{created_at}:{ttl_seconds}"
        signature = hmac.new(
            self._secret_key.encode("utf-8"),
            raw_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        token_str = f"STS-{secrets.token_urlsafe(24)}"
        return EphemeralToken(
            token=token_str,
            client_id=client_id,
            created_at=created_at,
            ttl_seconds=ttl_seconds,
            signature=signature
        )

    def validate_token(self, token_obj: EphemeralToken) -> bool:
        if token_obj.token in self._revoked_tokens:
            return False

        now = time.time()
        if now - token_obj.created_at > token_obj.ttl_seconds:
            return False  # Token expired

        raw_payload = f"{token_obj.client_id}:{token_obj.created_at}:{token_obj.ttl_seconds}"
        expected_sig = hmac.new(
            self._secret_key.encode("utf-8"),
            raw_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(token_obj.signature, expected_sig)

    def revoke_token(self, token_str: str) -> None:
        self._revoked_tokens.add(token_str)
