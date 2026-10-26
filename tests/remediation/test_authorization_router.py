import pytest
from core.remediation.authorization_router import (
    AuthoritativeAuthorizationRouter,
    FiduciaryAuthorityError,
    ModelAuthorizationError,
)

def test_allowed_model_authorization() -> None:
    router = AuthoritativeAuthorizationRouter()
    canonical = router.check_model_authorization("gpt-4o")
    assert canonical == "gpt-4o"

def test_denied_model_blocked() -> None:
    router = AuthoritativeAuthorizationRouter()
    with pytest.raises(ModelAuthorizationError, match=r"BLOCKED by quarantine policy"):
        router.check_model_authorization("blocked-legacy-model")

def test_denied_alias_blocked() -> None:
    router = AuthoritativeAuthorizationRouter()
    with pytest.raises(ModelAuthorizationError, match=r"BLOCKED by quarantine policy"):
        router.check_model_authorization("experimental-alias")

def test_tool_call_with_blocked_fallback_fails() -> None:
    router = AuthoritativeAuthorizationRouter()
    with pytest.raises(ModelAuthorizationError):
        router.execute_tool_call(
            model_identifier="gpt-4o",
            tool_name="calculate_covenant",
            tool_args={},
            fallback_models=["blocked-legacy-model"],
        )

def test_fiduciary_boundary_enforcement() -> None:
    router = AuthoritativeAuthorizationRouter()
    with pytest.raises(FiduciaryAuthorityError, match=r"Model consensus score .* cannot grant fiduciary authority"):
        router.assert_fiduciary_boundary(consensus_score=0.99, action="GRANT_CAPITAL")
