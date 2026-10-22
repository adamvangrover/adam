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
    mcp_evaluate_pdil_gatekeeper,
    mcp_score_t_plus_1_decision,
    eve_v2_underwriting_engine,
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

def test_mcp_evaluate_pdil_gatekeeper():
    import hashlib
    data_payload = {"ebitda": 100.0, "total_debt": 300.0}
    payload_bytes = json.dumps(data_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    content_hash = hashlib.sha256(payload_bytes).hexdigest()

    inference = {
        "data": data_payload,
        "provenance_trace": {
            "git_commit_hash": "abc123def456",
            "timestamp": "2026-08-26T12:00:00Z",
            "content_hash": content_hash,
            "jsonLogic_version": "1.0.0",
            "confidence_score": 0.95,
            "derivation_path": "ingest -> parse",
            "source_data_object": "SEC_EDGAR_MSFT_10K_EXHIBIT_99"
        }
    }
    rules = {"==": [{"var": "ebitda"}, 100.0]}
    res = json.loads(mcp_evaluate_pdil_gatekeeper(inference, rules=rules))
    assert res["status"] == "PASS"

def test_mcp_score_t_plus_1_decision():
    pred_trace = ["init", "retrieve", "reason", "validate", "synthesize", "complete"]
    gt_trace = ["init", "retrieve", "reason", "validate", "synthesize", "complete"]
    res = json.loads(mcp_score_t_plus_1_decision(pred_trace, gt_trace, [0.9, 0.95], [1, 1]))
    assert res["status"] == "PASS"
    assert res["trace_fidelity"] == 1.0

def test_mcp_eve_v2_underwriting_engine():
    res = json.loads(eve_v2_underwriting_engine(ticker="MSFT", spot_price=420.0))
    assert res["ticker"] == "MSFT"
    assert "final_eve_corridor_price" in res
    assert "verdict" in res
    assert "val_clocks" in res
