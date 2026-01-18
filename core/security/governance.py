import logging
import ast
import time
import hashlib
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class GovernanceError(Exception):
    """Raised when a governance policy is violated."""
    pass

class ApprovalRequired(GovernanceError):
    """Raised when an action requires human approval."""
    pass

class GovernanceEnforcer:
    """
    Enforces governance policies on code execution.
    Implements:
    - Conviction Scoring (Risk Assessment)
    - Dead Hand / Circuit Breakers
    - Human-in-the-Loop triggers
    """

    # Risk weights for different code patterns
    RISK_WEIGHTS = {
        'import': 10.0,      # Imports are generally risky in sandboxes
        'loop': 5.0,         # Infinite loop potential
        'recursion': 5.0,    # Recursion depth risk
        'complexity': 0.1,   # Per-node complexity penalty
        'dunder': 20.0,      # Accessing __private attributes
        'eval': 100.0,       # Explicit eval/exec is critical
    }

    # Thresholds
    AUTO_APPROVE_THRESHOLD = 20.0
    REQUIRE_APPROVAL_THRESHOLD = 50.0
    CRITICAL_BLOCK_THRESHOLD = 80.0

    @classmethod
    def analyze_risk(cls, code: str) -> Dict[str, Any]:
        """
        Analyzes code and returns a risk score and profile.
        """
        score = 0.0
        details = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"score": 100.0, "reason": "Syntax Error - potential obfuscation", "status": "BLOCKED"}

        # complexity = 0
        for node in ast.walk(tree):
            # complexity += 1
            node_type = type(node).__name__

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                score += cls.RISK_WEIGHTS['import']
                details.append("Import detected")

            if isinstance(node, (ast.For, ast.While)):
                score += cls.RISK_WEIGHTS['loop']
                details.append("Loop detected")

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Simple check for recursion (name in body)
                # This is weak but serves as a signal
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name) and subnode.func.id == node.name:
                        score += cls.RISK_WEIGHTS['recursion']
                        details.append("Recursion detected")
                        break

            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                score += cls.RISK_WEIGHTS['dunder']
                details.append(f"Private attribute access: {node.attr}")

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {'eval', 'exec', 'compile'}:
                    score += cls.RISK_WEIGHTS['eval']
                    details.append(f"Dangerous function: {node.func.id}")

            # Basic complexity penalty
            score += cls.RISK_WEIGHTS['complexity']

        # Status determination
        if score >= cls.CRITICAL_BLOCK_THRESHOLD:
            status = "BLOCKED"
        elif score >= cls.REQUIRE_APPROVAL_THRESHOLD:
            status = "APPROVAL_REQUIRED"
        else:
            status = "APPROVED"

        return {
            "score": round(score, 2),
            "status": status,
            "details": list(set(details))
        }

    @classmethod
    def validate(cls, code: str, context: str = "sandbox") -> bool:
        """
        Validates code against governance policies.
        Raises GovernanceError or ApprovalRequired if validation fails.
        """
        analysis = cls.analyze_risk(code)
        score = analysis["score"]
        status = analysis["status"]

        # Log the audit trail
        audit_entry = {
            "timestamp": time.time(),
            "context": context,
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "score": score,
            "status": status,
            "details": analysis["details"]
        }
        logger.info(f"Governance Audit: {audit_entry}")

        if status == "BLOCKED":
            raise GovernanceError(f"High risk code blocked (Score: {score}). Reasons: {analysis['details']}")

        if status == "APPROVAL_REQUIRED":
            # In a real system, this would trigger a workflow.
            # Here we raise an exception to simulate the "Stop the Line" behavior.
            raise ApprovalRequired(f"Code execution paused for Human-in-the-Loop approval (Score: {score}). Reasons: {analysis['details']}")

        return True
