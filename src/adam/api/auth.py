import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
# Conditional import to allow file to exist even if deps missing (for checking)
try:
    from joserfc import jwt
    from joserfc.errors import JoseError
    from joserfc.jwk import OctKey
    import base64

    def to_base64url(s: str) -> str:
        return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
except ImportError:
    jwt = None
    OctKey = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

import secrets
from typing import Optional, Any

# Ephemeral in-memory fallback key for open-source development sessions
# Generated dynamically per runtime process without hardcoded static secrets.
_OPEN_SOURCE_EPHEMERAL_KEY = secrets.token_hex(32)

def is_production_environment() -> bool:
    """Returns True if running in a hardened production environment."""
    env = os.environ.get("ENV", os.environ.get("AFOS_ENV", "development")).lower()
    return env in ("production", "prod")

def get_public_key() -> Any:
    """
    Zero-Trust public key resolver with open-source development fallback.
    - Production: Requires explicit JWT_PUBLIC_KEY (asymmetric) or JWT_SECRET_KEY, failing closed otherwise.
    - Open-Source / Local Dev: Gracefully falls back to public key or ephemeral development key.
    """
    # 1. Production enforcement: Fail closed
    if is_production_environment():
        if not os.environ.get("JWT_PUBLIC_KEY") and not os.environ.get("JWT_SECRET_KEY"):
            raise RuntimeError(
                "Zero-Trust Production Violation: JWT_PUBLIC_KEY or JWT_SECRET_KEY must be "
                "explicitly configured in production environments. Static fallback secrets are strictly prohibited."
            )

    if OctKey is None:
        raise RuntimeError("joserfc library is required for JWT key processing.")

    # 1. Asymmetric public key from environment (preferred in zero-trust)
    jwt_public_key = os.environ.get("JWT_PUBLIC_KEY")
    if jwt_public_key:
        try:
            if jwt_public_key.strip().startswith("{"):
                import json
                key_dict = json.loads(jwt_public_key)
                return OctKey.import_key(key_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to parse JWT_PUBLIC_KEY: {e}")

    # 2. Symmetric secret key from environment
    secret_key = os.environ.get("JWT_SECRET_KEY")
    if secret_key:
        return OctKey.import_key({
            "kty": "oct",
            "k": to_base64url(secret_key)
        })

    # 3. Open-source development graceful fallback (ephemeral in-memory key)
    return OctKey.import_key({
        "kty": "oct",
        "k": to_base64url(_OPEN_SOURCE_EPHEMERAL_KEY)
    })

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Validate Bearer token using joserfc.
    """
    if jwt is None:
        # Fallback if joserfc not installed (safety)
        raise HTTPException(status_code=500, detail="joserfc not installed")

    try:
        # Decode and validate signature
        # Note: In production, use your IdP's public key
        key = get_public_key()
        claims = jwt.decode(token, key)
        return claims.claims
    except JoseError as e:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
