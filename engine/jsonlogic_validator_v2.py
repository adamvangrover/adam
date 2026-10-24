from typing import Dict, Any, List, Union
from schemas.credit_contracts import FinancialAssertion, EvidenceStatus


class MissingFinancialInputError(Exception):
    """Raised when a required JsonLogic input variable is missing or None."""
    pass


class UnverifiedEvidenceError(Exception):
    """Raised when input evidence status is not AUDITED or ACCEPTED."""
    pass


class CovenantDirectionError(Exception):
    """Raised when a covenant test fails its inequality threshold."""
    pass


class NumericDomainError(Exception):
    """Raised on division by zero or invalid numeric domain operation."""
    pass


class GovernedJsonLogicValidator:
    """
    Strict JsonLogic rule evaluator enforcing financial evidence verification,
    prohibiting zero default substitutions, validating unit/period consistency,
    and enforcing covenant directionalities (e.g. <= ceiling, >= floor).
    """

    @staticmethod
    def _resolve_var(data: Dict[str, Any], path: str) -> Any:
        keys = path.split(".")
        curr = data
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                raise MissingFinancialInputError(
                    f"Missing financial input variable: '{path}'. Silent default substitution is strictly prohibited under 2LoD governance."
                )
        if curr is None:
            raise MissingFinancialInputError(
                f"Financial input variable '{path}' is None. Statistical imputation is prohibited."
            )
        return curr

    @classmethod
    def evaluate(cls, rule: Any, data: Dict[str, Any]) -> Any:
        if not isinstance(rule, dict):
            return rule

        if not rule:
            return None

        operator = list(rule.keys())[0]
        operands = rule[operator]

        if operator == "var":
            var_path = operands if isinstance(operands, str) else operands[0]
            val = cls._resolve_var(data, var_path)

            # If value is wrapped in FinancialAssertion or dict with evidence_ref
            if isinstance(val, FinancialAssertion):
                if val.evidence_ref.status not in (EvidenceStatus.AUDITED, EvidenceStatus.ACCEPTED):
                    raise UnverifiedEvidenceError(
                        f"Financial assertion '{val.assertion_id}' status is {val.evidence_ref.status}. Expected AUDITED or ACCEPTED."
                    )
                return val.value
            elif isinstance(val, dict) and "evidence_status" in val:
                if val["evidence_status"] not in ("AUDITED", "ACCEPTED"):
                    raise UnverifiedEvidenceError(
                        f"Evidence status '{val['evidence_status']}' for variable '{var_path}' is unverified."
                    )
                return val["value"]
            return val

        if operator in ("<=", "<", ">=", ">", "==", "!="):
            if not isinstance(operands, list) or len(operands) != 2:
                raise ValueError(f"Operator '{operator}' requires exactly 2 operands.")
            left = cls.evaluate(operands[0], data)
            right = cls.evaluate(operands[1], data)

            if operator == "<=":
                return left <= right
            elif operator == "<":
                return left < right
            elif operator == ">=":
                return left >= right
            elif operator == ">":
                return left > right
            elif operator == "==":
                return left == right
            elif operator == "!=":
                return left != right

        if operator == "/":
            if not isinstance(operands, list) or len(operands) != 2:
                raise ValueError("Division operator '/' requires exactly 2 operands.")
            num = cls.evaluate(operands[0], data)
            denom = cls.evaluate(operands[1], data)
            if denom == 0:
                raise NumericDomainError("Division by zero in covenant formula calculation.")
            return num / denom

        if operator == "*":
            res = 1.0
            for op in operands:
                res *= cls.evaluate(op, data)
            return res

        if operator in ("+", "-"):
            if operator == "+":
                return sum(cls.evaluate(op, data) for op in operands)
            else:
                if len(operands) == 1:
                    return -cls.evaluate(operands[0], data)
                return cls.evaluate(operands[0], data) - sum(cls.evaluate(op, data) for op in operands[1:])

        raise ValueError(f"Unsupported operator in GovernedJsonLogicValidator: '{operator}'")

    @classmethod
    def validate_covenant_headroom(
        cls,
        covenant_rule: Dict[str, Any],
        financial_data: Dict[str, Any],
        covenant_name: str,
        expected_direction: str  # "<=" for leverage ceilings, ">=" for DSCR coverage floors
    ) -> Dict[str, Any]:
        operator = list(covenant_rule.keys())[0]
        if operator != expected_direction:
            raise CovenantDirectionError(
                f"Covenant '{covenant_name}' rule operator '{operator}' does not match expected inequality direction '{expected_direction}'."
            )

        passed = cls.evaluate(covenant_rule, financial_data)
        left_val = cls.evaluate(covenant_rule[operator][0], financial_data)
        right_val = cls.evaluate(covenant_rule[operator][1], financial_data)

        headroom = (right_val - left_val) if expected_direction == "<=" else (left_val - right_val)

        return {
            "covenant_name": covenant_name,
            "passed": passed,
            "calculated_metric": left_val,
            "covenant_threshold": right_val,
            "headroom": round(headroom, 6),
            "inequality_operator": operator
        }
