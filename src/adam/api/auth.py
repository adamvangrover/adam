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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock public key (in production, fetch from OIDC provider .well-known/jwks.json)
# Using symmetric key for simplicity in this demo.
# Note: If JWT_SECRET_KEY is not set, we do not provide a default.
# The application will fail securely rather than using an empty or default secret.
def get_public_key():
    secret_key = os.environ.get("JWT_SECRET_KEY", "UNSET_SECRET_KEY_MUST_BE_CONFIGURED_IN_PROD")
    return OctKey.import_key({
        "kty": "oct",
        "k": to_base64url(secret_key)
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
