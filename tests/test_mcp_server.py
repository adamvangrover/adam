import json
import pytest
from server.mcp_server import (
    financial_statement_parser,
    divergence_validator,
    covenant_headroom_calculator,
    prov_o_audit_logger,
    model_alpha_client,
    model_beta_client,
    arbitration_calculator,
    audit_logger,
)

def test_mcp_server_financial_statement_parser():
    res = financial_statement_parser("MSFT", "10-K", "EBITDA")
    data = json.loads(res)
    assert data["entity_ticker"] == "MSFT"
    assert data["line_item"] == "EBITDA"
    assert len(data["prov_o_audit_trail"]) == 1

def test_mcp_server_divergence_validator():
    res = divergence_validator(0.04, 0.08, 0.03)
    data = json.loads(res)
    assert data["divergence"] == 0.04
    assert data["exceeds_threshold"] is True

def test_mcp_server_covenant_headroom_calculator():
    res = covenant_headroom_calculator(100.0, 400.0, 20.0, "ruleset-456")
    data = json.loads(res)
    assert data["fccr"] == 5.0
    assert data["leverage"] == 4.0
    assert data["covenant_ruleset_id"] == "ruleset-456"

def test_mcp_server_prov_o_audit_logger():
    res = prov_o_audit_logger("agent-alpha", "underwrite", 0.95, ["citation-1"])
    data = json.loads(res)
    assert data["agent_id"] == "agent-alpha"
    assert data["status"] == "committed"
    assert len(data["audit_hash"]) == 64

def test_mcp_server_credit_models():
    res_alpha = json.loads(model_alpha_client("AAPL", {"ebitda": 100.0, "total_debt": 200.0}))
    res_beta = json.loads(model_beta_client("AAPL", {"ebitda": 100.0, "total_debt": 200.0, "fccr": 2.0}))
    res_arb = json.loads(arbitration_calculator(res_alpha["pd"], res_beta["pd"]))

    assert "pd" in res_alpha
    assert "pd" in res_beta
    assert "consensus_pd" in res_arb

def test_mcp_server_audit_logger():
    res = json.loads(audit_logger("risk-guard", "policy_check", {"tier": "P0"}))
    assert res["status"] == "logged"
    assert res["agent_id"] == "risk-guard"
