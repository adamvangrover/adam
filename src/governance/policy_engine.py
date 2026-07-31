"""
Adam OS - Deterministic Financial Policy & Governance Engine.

This module provides a replay-safe, mathematically verifiable evaluation engine
using jsonLogic for financial covenants and OPA for authorization.
"""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
import structlog
from json_logic import jsonLogic  # type: ignore

logger = structlog.get_logger(__name__)

class FinancialContext(BaseModel):
    """Immutable context representing the current state of a financial entity."""
    entity_id: str
    asset_value_usd: float = Field(..., gt=0)
    total_debt_usd: float = Field(..., ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def ltv(self) -> float:
        """Calculates current Loan-to-Value. Formula: LTV = Debt / Asset"""
        return self.total_debt_usd / self.asset_value_usd

class PolicyResult(BaseModel):
    """Deterministic output of a policy evaluation."""
    passed: bool
    covenant_name: str
    evaluated_value: float
    threshold_value: float
    alert_triggered: bool
    details: str

class DeterministicPolicyEngine:
    """
    Stateless engine for evaluating financial rules safely within Temporal Activities.
    """

    def __init__(self, ruleset: Dict[str, Any]):
        self.ruleset = ruleset

    def evaluate_covenant(
        self,
        covenant_name: str,
        context: FinancialContext
    ) -> PolicyResult:
        """
        Evaluates a specific financial covenant deterministically.
        """
        if covenant_name not in self.ruleset:
            logger.error("missing_covenant_rule", covenant_name=covenant_name)
            raise ValueError(f"Covenant '{covenant_name}' not found in ruleset.")

        rule = self.ruleset[covenant_name]["logic"]
        threshold = self.ruleset[covenant_name]["threshold"]

        # Prepare deterministic data payload
        data = {
            "ltv": context.ltv,
            "asset_value": context.asset_value_usd,
            "total_debt": context.total_debt_usd
        }

        try:
            # jsonLogic evaluation guarantees deterministic AST processing
            passed = jsonLogic(rule, data)

            alert = False
            details = "Covenant maintained."

            if not passed:
                alert = True
                details = "⚠ SYSTEM WARNING: Covenant breach detected. Reflexive death spiral risk."
                logger.warning(
                    "covenant_breach",
                    covenant=covenant_name,
                    entity=context.entity_id,
                    ltv=context.ltv,
                    threshold=threshold
                )
            else:
                logger.info(
                    "covenant_passed",
                    covenant=covenant_name,
                    entity=context.entity_id,
                    ltv=context.ltv
                )

            return PolicyResult(
                passed=bool(passed),
                covenant_name=covenant_name,
                evaluated_value=context.ltv,
                threshold_value=threshold,
                alert_triggered=alert,
                details=details
            )

        except Exception as e:
            logger.error("evaluation_failure", error=str(e), covenant=covenant_name)
            raise RuntimeError(f"Policy evaluation failed: {e}")

class OPAAuthorizationClient:
    """Mock integration for Open Policy Agent (OPA)."""

    @staticmethod
    def is_authorized(principal_role: str, action: str) -> bool:
        # In a real system, this would make an HTTP call to the OPA sidecar
        # e.g., POST http://localhost:8181/v1/data/adam_os/authz/allow
        logger.info("opa_eval_mock", role=principal_role, action=action)
        if principal_role == "sovereign_analyst" and action in ["evaluate_risk", "execute_lbo"]:
            return True
        return False
