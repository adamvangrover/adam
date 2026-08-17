"""
Standard assertions for the Evaluation Harness.
"""

def assert_invariant(condition: bool, failure_message: str):
    """
    Asserts a critical invariant. If false, raises an AssertionError.
    """
    if not condition:
        raise AssertionError(f"INVARIANT VIOLATION: {failure_message}")

def fail_closed_if(condition: bool, reason: str):
    """
    Helper to enforce fail-closed architecture requirements.
    """
    if condition:
        return {"status": "BLOCKED", "severity": "CRITICAL", "reason": reason}
    return {"status": "PASS"}
