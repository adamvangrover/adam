"""
Ephemeral Security Token Service (STS) with 15-minute max TTL.
Zero-Trust Architecture:
1. Short-lived ephemeral tokens (15-min / 900s rigid cap).
2. Asymmetric Public-Key verification capability (verifier does not need signing secrets).
3. Graceful fallback for open-source development and fail-closed security in production.
"""

import os
import time
import secrets
import hmac
import hashlib
from typing import Dict, Any, Optional

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class SecurityTokenError(Exception):
    """Raised when token validation fails."""
    pass


class SecurityTokenService:
    """
    Issues short-lived signed access tokens with strict TTL (max 15 mins / 900 seconds).
    Supports asymmetric zero-trust verification (verifiers need only the public key)
    with graceful fallback to ephemeral keys for open-source development.
    """

    MAX_TTL_SECONDS = 900

    def __init__(
        self,
        secret_key: Optional[str] = None,
        private_key: Optional[Any] = None,
        public_key: Optional[Any] = None,
        max_ttl_seconds: int = 900
    ):
        self.max_ttl_seconds = min(max_ttl_seconds, self.MAX_TTL_SECONDS)
        self.open_source_fallback = False

        is_prod = os.environ.get("ENV", os.environ.get("AFOS_ENV", "development")).lower() in ("production", "prod")

        self.private_key = private_key
        self.public_key = public_key

        if self.private_key is not None and HAS_CRYPTOGRAPHY and self.public_key is None:
            self.public_key = self.private_key.public_key()

        if secret_key:
            self.secret_key = secret_key
        elif self.private_key is not None or self.public_key is not None:
            self.secret_key = None
        else:
            if is_prod:
                raise SecurityTokenError(
                    "Zero-Trust Production Violation: Cryptographic key material (private_key, public_key, or secret_key) "
                    "must be configured in production environments. Static fallback secrets are strictly prohibited."
                )
            # Graceful fallback for open-source work: generate ephemeral in-memory keypair or secret
            self.open_source_fallback = True
            if HAS_CRYPTOGRAPHY:
                self.private_key = ec.generate_private_key(ec.SECP256R1())
                self.public_key = self.private_key.public_key()
                self.secret_key = None
            else:
                self.secret_key = secrets.token_hex(32)

    def issue_token(self, client_id: str, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Issues an ephemeral token for a client with an expiration timestamp.
        """
        effective_ttl = min(ttl_seconds or self.max_ttl_seconds, self.max_ttl_seconds)
        expires_at = int(time.time()) + effective_ttl
        payload_str = f"{client_id}:{expires_at}"

        if self.private_key is not None and HAS_CRYPTOGRAPHY:
            signature_bytes = self.private_key.sign(
                payload_str.encode("utf-8"),
                ec.ECDSA(hashes.SHA256())
            )
            token = f"{payload_str}:{signature_bytes.hex()}"
        elif self.secret_key is not None:
            signature = hmac.new(
                self.secret_key.encode("utf-8"),
                payload_str.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            token = f"{payload_str}:{signature}"
        else:
            raise SecurityTokenError("Cannot issue token: No private key or signing secret available.")

        return {
            "token": token,
            "client_id": client_id,
            "expires_at": expires_at,
            "ttl_seconds": effective_ttl,
            "open_source_fallback": self.open_source_fallback,
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

        payload_str = f"{client_id}:{expires_at_str}"

        # 1. Try Asymmetric Public-Key validation if public_key is available
        if self.public_key is not None and HAS_CRYPTOGRAPHY:
            try:
                sig_bytes = bytes.fromhex(signature)
                self.public_key.verify(
                    sig_bytes,
                    payload_str.encode("utf-8"),
                    ec.ECDSA(hashes.SHA256())
                )
                return {
                    "client_id": client_id,
                    "expires_at": expires_at,
                    "valid": True,
                    "verification_mode": "asymmetric_public_key"
                }
            except Exception:
                if self.secret_key is None:
                    raise SecurityTokenError("Invalid STS token signature (asymmetric verification failed).")

        # 2. Try Symmetric HMAC validation if secret_key is available
        if self.secret_key is not None:
            expected_signature = hmac.new(
                self.secret_key.encode("utf-8"),
                payload_str.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()

            if hmac.compare_digest(signature, expected_signature):
                return {
                    "client_id": client_id,
                    "expires_at": expires_at,
                    "valid": True,
                    "verification_mode": "symmetric_hmac"
                }

        raise SecurityTokenError("Invalid STS token signature.")
