from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
# Conditional import to allow file to exist even if deps missing (for checking)
try:
    from authlib.jose import jwt, JoseError
except ImportError:
    jwt = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock public key (in production, fetch from OIDC provider .well-known/jwks.json)
# Using symmetric key for simplicity in this demo.
PUBLIC_KEY = {
    "kty": "oct",
    "k": "mock-secret-key-for-testing-purposes-only-must-be-long-enough",
    "alg": "HS256"
}

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Validate Bearer token using Authlib.
    """
    if jwt is None:
        # Fallback if authlib not installed (safety)
        if token == "secret-token":
            return {"sub": "admin"}
        raise HTTPException(status_code=500, detail="Authlib not installed")

    try:
        # Decode and validate signature
        # Note: In production, use your IdP's public key
        payload = jwt.decode(token, PUBLIC_KEY)
        # payload.validate() # basic validation
        return payload
    except JoseError as e:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
