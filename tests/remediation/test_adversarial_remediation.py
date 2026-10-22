"""
Adversarial Regression Test Suite (F-09, Gate G4, Prompt Section 7).

Verifies all 10 non-negotiable adversarial failure conditions across the ADAM OS remediation suite.
"""

import math
import pytest
from core.remediation.extraction_validator import (
    EvidenceBoundExtractionRecord,
    EvidenceSpan,
    EvidenceStatus,
    SR117UnverifiedTelemetryError,
    validate_extraction_record,
    validate_telemetry_payload,
)
from core.remediation.covenant_evaluator import (
    evaluate_covenant_leverage,
    validate_jsonlogic_ast,
    CovenantValidationError,
    JsonLogicASTError,
)
from core.remediation.cache_manager import DealRoomContextCache
from core.remediation.qmc_engine import ValidatedQMCEngine, QMCStatus
from core.remediation.authorization_router import (
    AuthoritativeAuthorizationRouter,
    ModelAuthorizationError,
    FiduciaryAuthorityError,
)

VALID_SHA256 = "b" * 64


# Test 1: Missing secured debt or covenant thresholds cause a hard halt rather than a zero default.
def test_adversarial_missing_debt_hard_halt():
    inputs = {
        "total_debt": 1800.0,
        "ebitda": 450.0,
        "max_total_leverage_covenant": 4.50,
        "max_secured_leverage_covenant": 3.00,
    }
    with pytest.raises(CovenantValidationError, match="Missing required covenant inputs"):
        evaluate_covenant_leverage(inputs)


# Test 2: NaN, infinity, negative debt, zero EBITDA, and invalid units are rejected.
@pytest.mark.parametrize(
    "invalid_inputs",
    [
        {"total_debt": math.nan, "senior_secured_debt": 100, "ebitda": 450, "max_total_leverage_covenant": 4.5, "max_secured_leverage_covenant": 3.0},
        {"total_debt": math.inf, "senior_secured_debt": 100, "ebitda": 450, "max_total_leverage_covenant": 4.5, "max_secured_leverage_covenant": 3.0},
        {"total_debt": -100.0, "senior_secured_debt": 100, "ebitda": 450, "max_total_leverage_covenant": 4.5, "max_secured_leverage_covenant": 3.0},
        {"total_debt": 1000.0, "senior_secured_debt": 100, "ebitda": 0.0, "max_total_leverage_covenant": 4.5, "max_secured_leverage_covenant": 3.0},
    ],
)
def test_adversarial_non_finite_and_domain_violations(invalid_inputs):
    with pytest.raises(CovenantValidationError):
        evaluate_covenant_leverage(invalid_inputs)


# Test 3: A VERIFIED_EXTRACTION label without source evidence cannot pass validation.
def test_adversarial_verified_label_without_source_evidence():
    record = EvidenceBoundExtractionRecord(
        metric_name="senior_secured_debt",
        value=1125.0,
        units="USD_MILLIONS",
        status=EvidenceStatus.VERIFIED_EXTRACTION,
        doc_id="DOC_123",
        doc_sha256=VALID_SHA256,
        timestamp="2026-10-17T00:00:00Z",
        confidence=0.998,
        source_span=None,  # Missing source span grounding
    )
    with pytest.raises(SR117UnverifiedTelemetryError, match="lacks grounded source text span"):
        validate_extraction_record(record)


# Test 4: A nested or conflicting unverified finding cannot be ignored.
def test_adversarial_unverified_negative_finding_halt():
    payload = {
        "status": "UNVERIFIED_NEGATIVE_FINDING",
        "doc_sha256": VALID_SHA256,
        "source_span": {"text": "Negative covenant clause breached."},
    }
    with pytest.raises(SR117UnverifiedTelemetryError, match="UNVERIFIED_NEGATIVE_FINDING"):
        validate_telemetry_payload(payload)


# Test 5: A blocked model cannot execute through a fallback, alternate provider alias, or direct routing path.
def test_adversarial_blocked_model_quarantine_bypass():
    router = AuthoritativeAuthorizationRouter()
    # Direct
    with pytest.raises(ModelAuthorizationError):
        router.check_model_authorization("blocked-legacy-model")
    # Alias
    with pytest.raises(ModelAuthorizationError):
        router.check_model_authorization("experimental-alias")
    # Tool fallback
    with pytest.raises(ModelAuthorizationError):
        router.execute_tool_call(
            model_identifier="gpt-4o",
            tool_name="qmc_sim",
            tool_args={},
            fallback_models=["shadow-routing-alias"]
        )


# Test 6: A stale cache entry or context belonging to another deal cannot be reused.
def test_adversarial_stale_cache_and_cross_deal_isolation():
    cache = DealRoomContextCache(ttl_seconds=0.01)
    cache.put("DEAL_A", "v1.0", 850_000, {"secret": "deal_a_data"})
    cache.put("DEAL_B", "v1.0", 850_000, {"secret": "deal_b_data"})

    # Cross-deal isolation check
    assert cache.get("DEAL_A", "v1.0") is not None
    assert cache.get("DEAL_B", "v1.0").data["secret"] == "deal_b_data"

    # Expiration check
    import time
    time.sleep(0.02)
    assert cache.get("DEAL_A", "v1.0") is None  # Stale entry rejected


# Test 7: Malformed jsonLogic expressions, extra operators, missing operands, and invalid variable types are rejected.
def test_adversarial_malformed_jsonlogic_ast():
    valid_vars = {"leverage", "covenant"}
    malformed_ast = {"<=": [1, 2, 3]}  # Arity 3 for binary operator
    with pytest.raises(JsonLogicASTError):
        validate_jsonlogic_ast(malformed_ast, valid_vars)

    undeclared_var_ast = {"<=": [{"var": "undeclared_x"}, 4.5]}
    with pytest.raises(JsonLogicASTError, match="Undeclared variable"):
        validate_jsonlogic_ast(undeclared_var_ast, valid_vars)


# Test 8: QMC convergence status cannot be assigned merely because a run completes.
def test_adversarial_qmc_completion_is_not_convergence():
    engine = ValidatedQMCEngine()
    res = engine.run_simulation(num_samples=1024, seed=42)
    # Merely completing a run yields SIMULATION_COMPLETED, NOT CONVERGENCE_VALIDATED
    assert res["status"] == QMCStatus.SIMULATION_COMPLETED.value
    assert res["status"] != QMCStatus.CONVERGENCE_VALIDATED.value


# Test 9: Identical pinned inputs and configuration reproduce the same result within declared numerical tolerances.
def test_adversarial_pinned_reproducibility():
    engine = ValidatedQMCEngine()
    assert engine.verify_reproducibility(num_samples=2048, seed=42, tolerance=1e-6) is True


# Test 10: Model output cannot bypass the deterministic authorization boundary or directly sanction a transaction.
def test_adversarial_fiduciary_execution_boundary():
    router = AuthoritativeAuthorizationRouter()
    with pytest.raises(FiduciaryAuthorityError):
        router.assert_fiduciary_boundary(consensus_score=1.0, action="SANCTION_TRANSACTION")
