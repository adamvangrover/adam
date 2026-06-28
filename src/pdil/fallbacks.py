from typing import Dict, Any
from src.pdil.middleware import SecurityGovernanceGatekeeper, GovernanceError

class CircuitBreaker:
    """Redundant fallback to prevent cascading failures in deterministic systems."""
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.failures = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.open = True

    def reset(self):
        self.failures = 0
        self.open = False

class IndependentGatekeeperCheck:
    """
    Maintains the entire gatekeeper as a redundant, fallback modular independent check.
    If primary systems fail or require an additive verification layer, this executes the full governance check.
    """
    def __init__(self, schema: Dict[str, Any]):
        self.gatekeeper = SecurityGovernanceGatekeeper(schema)

    def perform_redundant_check(self, payload: Dict[str, Any]) -> bool:
        """
        Executes an independent validation pass as a complementary check.
        Returns True if valid, False if it violates governance (does not raise).
        """
        try:
            self.gatekeeper.validate_inference(payload)
            return True
        except GovernanceError:
            return False

__all__ = ["CircuitBreaker", "IndependentGatekeeperCheck"]
