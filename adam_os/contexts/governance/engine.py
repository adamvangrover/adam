from typing import Dict, Any
from pydantic import BaseModel
from json_logic import jsonLogic
import structlog

logger = structlog.get_logger()

class PolicyResult(BaseModel):
    is_breached: bool
    evaluation_details: Dict[str, Any]

class DeterministicPolicyEngine:
    """Deterministic policy engine that uses jsonLogic to evaluate financial contexts."""

    def __init__(self) -> None:
        # Rules defined in jsonLogic
        # LTV is breached if (debt / asset_value) > 0.35
        # DSCR is breached if (net_operating_income / debt_service) < 1.25
        self.rules = {
            "ltv_35": {
                ">": [
                    {"/": [{"var": "debt"}, {"var": "asset_value"}]},
                    0.35
                ]
            },
            "dscr_125": {
                "<": [
                    {"/": [{"var": "net_operating_income"}, {"var": "debt_service"}]},
                    1.25
                ]
            }
        }
        logger.info("initialized_deterministic_policy_engine", rules=list(self.rules.keys()))

    def evaluate(self, rule_name: str, context: Dict[str, Any]) -> PolicyResult:
        if rule_name not in self.rules:
            raise ValueError(f"Rule {rule_name} not found")

        rule = self.rules[rule_name]
        is_breached = jsonLogic(rule, context)

        logger.info("evaluated_rule", rule_name=rule_name, is_breached=is_breached)
        return PolicyResult(
            is_breached=is_breached,
            evaluation_details={"rule_name": rule_name, "context": context, "rule": rule}
        )
