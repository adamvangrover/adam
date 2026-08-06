import json
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
import structlog
from json_logic import jsonLogic

logger = structlog.get_logger(__name__)

class FinancialContext(BaseModel):
    """Immutable context representing the current state of a financial entity."""
    entity_id: str
    asset_value_usd: float = Field(..., gt=0)
    total_debt_usd: float = Field(..., ge=0)
    
    # Brought in from HEAD's DSCR requirement
    net_operating_income: Optional[float] = None
    debt_service: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def ltv(self) -> float:
        """Calculates current Loan-to-Value. Formula: LTV = Debt / Asset"""
        return self.total_debt_usd / self.asset_value_usd
        
    @property
    def dscr(self) -> Optional[float]:
        """Calculates Debt Service Coverage Ratio if data is available."""
        if self.net_operating_income is not None and self.debt_service:
            return self.net_operating_income / self.debt_service
        return None

class PolicyResult(BaseModel):
    """Deterministic output of a policy evaluation, merged from both branches."""
    passed: bool
    is_breached: bool
    covenant_name: str
    evaluated_value: Optional[float] = None
    threshold_value: Optional[float] = None
    alert_triggered: bool
    details: str
    evaluation_details: Dict[str, Any]

class DeterministicPolicyEngine:
    """
    Stateless, expandable engine for evaluating financial rules deterministically.
    Supports both strict Pydantic contexts and raw dictionaries for open accessibility.
    """
    def __init__(self, ruleset: Optional[Dict[str, Any]] = None):
        # Default rules combining logic from both branches if none are provided
        # Note: Logic is framed as "passing" conditions to align with main's implementation
        self.ruleset = ruleset or {
            "ltv_35": {
                "logic": {"<=": [{"var": "ltv"}, 0.35]},
                "threshold": 0.35
            },
            "dscr_125": {
                "logic": {">=": [{"var": "dscr"}, 1.25]},
                "threshold": 1.25
            }
        }
        logger.info("initialized_deterministic_policy_engine", rules=list(self.ruleset.keys()))

    def add_rule(self, name: str, logic: Dict[str, Any], threshold: Optional[float] = None) -> None:
        """
        Dynamically expand the engine by injecting new rules at runtime.
        """
        self.ruleset[name] = {"logic": logic, "threshold": threshold}
        logger.info("rule_added", rule_name=name)

    def evaluate(
        self,
        rule_name: str,
        context: Union[FinancialContext, Dict[str, Any]]
    ) -> PolicyResult:
        """
        Evaluates a specific financial rule deterministically against a context.
        """
        if rule_name not in self.ruleset:
            logger.error("missing_rule", rule_name=rule_name)
            raise ValueError(f"Rule '{rule_name}' not found in ruleset.")

        # Support both main's nested dict approach and HEAD's raw logic approach
        rule_def = self.ruleset[rule_name]
        rule_logic = rule_def.get("logic", rule_def) if isinstance(rule_def, dict) else rule_def
        threshold = rule_def.get("threshold") if isinstance(rule_def, dict) else None

        # Prepare deterministic data payload (accepts model or open dictionary)
        if isinstance(context, FinancialContext):
            data = context.model_dump()
            data['ltv'] = context.ltv
            data['dscr'] = context.dscr
            entity_id = context.entity_id
        else:
            data = context
            entity_id = data.get("entity_id", "unknown")

        try:
            # jsonLogic evaluation guarantees deterministic AST processing
            passed = jsonLogic(rule_logic, data)
            is_breached = not passed

            alert = is_breached
            details = "Covenant maintained." if passed else "⚠ SYSTEM WARNING: Covenant breach detected."

            # Audit logging setup
            log_kwargs = {
                "covenant": rule_name,
                "entity": entity_id,
                "threshold": threshold,
                "prov_o_audit_trail": {
                    "wasGeneratedBy": "DeterministicPolicyEngine",
                    "usedRulesetVersion": "v2_merged",
                    "derivation_path": "jsonLogic_evaluation",
                }
            }

            if is_breached:
                logger.warn("covenant_breach", **log_kwargs)
            else:
                logger.info("covenant_passed", **log_kwargs)
                
            # Attempt to extract the evaluated value for the payload if it maps cleanly
            eval_val = data.get("ltv") if "ltv" in rule_name.lower() else data.get("dscr")

            return PolicyResult(
                passed=bool(passed),
                is_breached=bool(is_breached),
                covenant_name=rule_name,
                evaluated_value=eval_val,
                threshold_value=threshold,
                alert_triggered=alert,
                details=details,
                evaluation_details={
                    "rule_name": rule_name,
                    "context_snapshot": data,
                    "rule_logic": rule_logic
                }
            )

        except Exception as e:
            logger.error("evaluation_failure", error=str(e), rule_name=rule_name)
            raise RuntimeError(f"Policy evaluation failed: {e}")