import json
import pytest
from servers.mcp_registry import (
    financial_statement_parser,
    divergence_validator,
    covenant_headroom_calculator,
    prov_o_audit_logger
)

def test_financial_statement_parser():
    res = financial_statement_parser("AAPL", "10-K", "EBITDA")
    data = json.loads(res)
    assert data["entity_ticker"] == "AAPL"
    assert "provenance" in data
    assert data["provenance"]["@context"] == "http://www.w3.org/ns/prov"

def test_divergence_validator():
    res = divergence_validator(0.05, 0.08, 0.02)
    data = json.loads(res)
    assert data["exceeds_threshold"] is True
    assert data["divergence"] == pytest.approx(0.03)

def test_covenant_headroom_calculator():
    res = covenant_headroom_calculator(100.0, 300.0, 10.0, "uuid-123")
    data = json.loads(res)
    assert data["fccr"] == 10.0
    assert data["leverage"] == 3.0
    assert data["covenant_ruleset_id"] == "uuid-123"

def test_prov_o_audit_logger():
    res = prov_o_audit_logger("agent-X", "Testing", 0.95, ["source1"])
    data = json.loads(res)
    assert data["status"] == "committed"
    assert "audit_hash" in data
