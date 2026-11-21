from hypothesis import given, strategies as st

from core.evaluation.integration_verifier import decide_integration

@given(
    confidence=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_failed_hard_gate_can_never_integrate(confidence: float):
    decision = decide_integration(
        hard_gate_passed=False,
        scope_passed=True,
        policy_passed=True,
        provenance_passed=True,
        calibration_healthy=True,
        confidence=confidence,
        threshold=0.0,
    )

    assert not decision.allowed

@given(
    confidence=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_scope_violation_can_never_integrate(confidence: float):
    decision = decide_integration(
        hard_gate_passed=True,
        scope_passed=False,
        policy_passed=True,
        provenance_passed=True,
        calibration_healthy=True,
        confidence=confidence,
        threshold=0.0,
    )

    assert not decision.allowed

@given(
    confidence=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_provenance_failure_can_never_integrate(confidence: float):
    decision = decide_integration(
        hard_gate_passed=True,
        scope_passed=True,
        policy_passed=True,
        provenance_passed=False,
        calibration_healthy=True,
        confidence=confidence,
        threshold=0.0,
    )

    assert not decision.allowed
