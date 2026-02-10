from core.engine.meta_orchestrator import MetaOrchestrator
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import secrets
from core.settings import settings

_orchestrator = None


def get_orchestrator() -> MetaOrchestrator:
    """
    Returns a singleton instance of the MetaOrchestrator.
    """
    global _orchestrator
    if _orchestrator is None:
        # Initialize with default legacy orchestrator handled internally by MetaOrchestrator
        _orchestrator = MetaOrchestrator()
    return _orchestrator

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verifies the API Key from the X-API-Key header.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )

    if not secrets.compare_digest(api_key, settings.adam_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key
