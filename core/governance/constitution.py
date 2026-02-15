from dataclasses import dataclass, field
from typing import List, Dict, Any
import time
from core.utils.logger import setup_logger

logger = setup_logger("GovernanceConstitution")

@dataclass
class Principle:
    id: str
    description: str
    weight: float = 1.0

class Constitution:
    """
    Defines the core principles and rules for agent behavior.
    Acts as a governance layer for all actions.
    """
    def __init__(self):
        self.principles: List[Principle] = [
            Principle("DO_NO_HARM", "Do not take actions that cause financial ruin or systemic risk.", 10.0),
            Principle("TRANSPARENCY", "All actions must be explainable and logged.", 5.0),
            Principle("COMPLIANCE", "Adhere to all regulatory frameworks (e.g., SEC, GDPR).", 8.0),
            Principle("RISK_MANAGEMENT", "Maintain risk exposure within defined limits.", 7.0)
        ]
        self.audit_log: List[Dict[str, Any]] = []

    def check_action(self, action: str, context: Dict[str, Any]) -> bool:
        """
        Validates an action against the constitution.

        Args:
            action (str): The name of the action (e.g., "EXECUTE_TRADE").
            context (dict): Contextual information (e.g., {"amount": 1000, "risk_score": 0.2}).

        Returns:
            bool: True if the action is allowed, False otherwise.
        """
        logger.info(f"Checking action: {action} with context: {context}")

        # Simple heuristic checks (mock implementation)
        # 1. DO_NO_HARM / RISK_MANAGEMENT
        risk_score = context.get("risk_score", 0.0)
        if risk_score > 0.8:
            logger.warning(f"Action {action} blocked due to high risk score: {risk_score}")
            self._log_audit(action, context, False, "Risk score exceeds threshold.")
            return False

        # 2. COMPLIANCE
        if "illegal" in action.lower():
            logger.warning(f"Action {action} blocked due to compliance violation.")
            self._log_audit(action, context, False, "Action appears illegal.")
            return False

        logger.info(f"Action {action} approved.")
        self._log_audit(action, context, True, "Action complies with constitution.")
        return True

    def _log_audit(self, action: str, context: Dict[str, Any], allowed: bool, reason: str):
        entry = {
            "timestamp": time.time(),
            "action": action,
            "context": context,
            "allowed": allowed,
            "reason": reason
        }
        self.audit_log.append(entry)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return self.audit_log
