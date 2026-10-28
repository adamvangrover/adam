"""
Ephemeral Security Token Service (STS) with 15-minute max TTL.
"""

import time
import secrets
import hmac
import hashlib
from typing import Dict, Any, Optional


class SecurityTokenError(Exception):
    """Raised when token validation fails."""
    pass


class SecurityTokenService:
    """
    Issues short-lived HMAC-SHA256 signed access tokens with strict TTL (max 15 mins / 900 seconds).
    """

    def __init__(self, secret_key: Optional[str] = None, max_ttl_seconds: int = 900):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.max_ttl_seconds = min(max_ttl_seconds, 900)  # Rigid cap at 15 minutes

    def issue_token(self, client_id: str, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Issues an ephemeral token for a client with an expiration timestamp.
        """
        effective_ttl = min(ttl_seconds or self.max_ttl_seconds, self.max_ttl_seconds)
        expires_at = int(time.time()) + effective_ttl

        payload_str = f"{client_id}:{expires_at}"
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        token = f"{payload_str}:{signature}"
        return {
            "token": token,
            "client_id": client_id,
            "expires_at": expires_at,
            "ttl_seconds": effective_ttl,
        }

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validates token signature and expiration.
        """
        try:
            parts = token.split(":")
            if len(parts) != 3:
                raise SecurityTokenError("Malformed STS token structure.")

            client_id, expires_at_str, signature = parts
            expires_at = int(expires_at_str)
        except Exception as e:
            raise SecurityTokenError(f"Invalid STS token format: {e}")

        # Check expiration
        now = int(time.time())
        if now > expires_at:
            raise SecurityTokenError(f"STS token expired ({now - expires_at}s ago). Max TTL is 15 minutes.")

        # Verify signature
        payload_str = f"{client_id}:{expires_at_str}"
        expected_signature = hmac.new(
            self.secret_key.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            raise SecurityTokenError("Invalid STS token signature.")

        return {
            "client_id": client_id,
            "expires_at": expires_at,
            "valid": True,
        }
