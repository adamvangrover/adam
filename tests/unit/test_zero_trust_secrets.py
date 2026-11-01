import os
import time
import pytest
from unittest.mock import patch

from src.zero_trust.sts import SecurityTokenService, SecurityTokenError
from src.adam.api.auth import is_production_environment, get_public_key


def test_sts_open_source_fallback_lifecycle():
    # In non-production, STS should gracefully fall back to ephemeral development keys
    sts = SecurityTokenService(max_ttl_seconds=900)
    assert sts.open_source_fallback is True

    # Issue token
    res = sts.issue_token(client_id="open-source-node", ttl_seconds=300)
    assert res["ttl_seconds"] == 300
    assert "token" in res

    # Validate token
    claims = sts.validate_token(res["token"])
    assert claims["client_id"] == "open-source-node"
    assert claims["valid"] is True


def test_sts_max_ttl_enforcement():
    # Rigid cap at 900 seconds (15 minutes)
    sts = SecurityTokenService(max_ttl_seconds=1800)
    assert sts.max_ttl_seconds == 900

    res = sts.issue_token(client_id="node-test", ttl_seconds=5000)
    assert res["ttl_seconds"] == 900


def test_sts_expired_token_rejection():
    sts = SecurityTokenService()
    expired_token = f"node-test:{int(time.time()) - 50}:invalidsig"
    with pytest.raises(SecurityTokenError, match="expired"):
        sts.validate_token(expired_token)


def test_sts_production_fail_closed():
    # In production environments, missing cryptographic key material must fail closed
    with patch.dict(os.environ, {"AFOS_ENV": "production"}):
        with pytest.raises(SecurityTokenError, match="Zero-Trust Production Violation"):
            SecurityTokenService()


def test_sts_symmetric_explicit_key():
    # Providing an explicit secret key works in both production and development
    sts = SecurityTokenService(secret_key="my_custom_secret_key_123")
    assert sts.open_source_fallback is False

    res = sts.issue_token(client_id="service-a")
    claims = sts.validate_token(res["token"])
    assert claims["client_id"] == "service-a"
    assert claims["valid"] is True


def test_auth_is_production_environment():
    with patch.dict(os.environ, {"ENV": "production"}):
        assert is_production_environment() is True

    with patch.dict(os.environ, {"AFOS_ENV": "prod"}):
        assert is_production_environment() is True

    with patch.dict(os.environ, {"ENV": "development"}):
        assert is_production_environment() is False


def test_auth_production_fail_closed():
    with patch.dict(os.environ, {"ENV": "production"}, clear=True):
        # When neither JWT_PUBLIC_KEY nor JWT_SECRET_KEY is set in prod
        try:
            get_public_key()
            pytest.fail("Expected get_public_key() to raise RuntimeError in production")
        except RuntimeError as e:
            assert "Zero-Trust Production Violation" in str(e) or "joserfc library is required" in str(e)
