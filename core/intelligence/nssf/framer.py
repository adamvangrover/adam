import ast
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from core.schemas.v23_5_schema import ExecutionPlan

logger = logging.getLogger(__name__)

class RefinedExecutionPlan(ExecutionPlan):
    """
    Extends the standard ExecutionPlan with deterministic probability overrides
    and NSSF meta-data.
    """
    collapser_verdict: str = Field(..., description="Final deterministic state: EXECUTE, ABORT, or HOLD")
    original_probability: float = Field(..., description="Raw probability from Liquid Network")
    applied_constraints: List[str] = Field(default_factory=list, description="Rules that triggered")

class ConstraintRule(BaseModel):
    rule_id: str
    description: str
    condition_code: str # Python code string for evaluation
    action: str # "FORCE_0", "FORCE_1", "ABORT", etc.

class DeterministicProbabilityScript:
    """
    Manages the Hard Constraints that overlay the probabilistic output.
    """
    def __init__(self):
        self.rules: List[ConstraintRule] = []
        self._load_default_rules()

    def _load_default_rules(self):
        # Example Hard Logic
        self.rules.append(ConstraintRule(
            rule_id="VOLATILITY_GATE",
            description="Prevent trade if volatility is extreme",
            condition_code="context.get('volatility', 0) > 0.5",
            action="FORCE_0"
        ))
        self.rules.append(ConstraintRule(
            rule_id="HIGH_CONFIDENCE_ACCELERATOR",
            description="If liquid confidence is high and uncertainty low, force execution",
            condition_code="liquid_score > 0.8 and context.get('uncertainty', 1.0) < 0.2",
            action="FORCE_1"
        ))

        # New Financial Rules
        self.rules.append(ConstraintRule(
            rule_id="LIQUIDITY_CRUNCH_ABORT",
            description="Abort if liquidity risk is High",
            condition_code="context.get('liquidity_risk') == 'High'",
            action="FORCE_0"
        ))

        self.rules.append(ConstraintRule(
            rule_id="WEAK_COLLATERAL_HOLD",
            description="Force HOLD if collateral coverage is Weak but other signals okay",
            condition_code="context.get('collateral_coverage') == 'Weak' and liquid_score > 0.5",
            action="FORCE_0" # Effectively Hold/Abort depending on interpretation, here 0 = Abort/Hold
        ))

    def _safe_eval_condition(self, node, eval_context):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in eval_context:
                return eval_context[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.Compare):
            left_val = self._safe_eval_condition(node.left, eval_context)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right_val = self._safe_eval_condition(comparator, eval_context)
                res = False
                if isinstance(op, ast.Gt):
                    res = left_val > right_val
                elif isinstance(op, ast.Lt):
                    res = left_val < right_val
                elif isinstance(op, ast.GtE):
                    res = left_val >= right_val
                elif isinstance(op, ast.LtE):
                    res = left_val <= right_val
                elif isinstance(op, ast.Eq):
                    res = left_val == right_val
                elif isinstance(op, ast.NotEq):
                    res = left_val != right_val
                else:
                    raise ValueError(f"Unsupported operator: {type(op)}")
                if not res:
                    return False
                left_val = right_val
            return True
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self._safe_eval_condition(node.operand, eval_context)
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not self._safe_eval_condition(value, eval_context):
                        return False
                return True
            elif isinstance(node.op, ast.Or):
                for value in node.values:
                    if self._safe_eval_condition(value, eval_context):
                        return True
                return False
            else:
                raise ValueError(f"Unsupported boolop: {type(node.op)}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "context" and node.func.attr == "get":
                    if len(node.args) < 1 or len(node.args) > 2:
                        raise ValueError("context.get requires 1 or 2 arguments")
                    key = self._safe_eval_condition(node.args[0], eval_context)
                    default = self._safe_eval_condition(node.args[1], eval_context) if len(node.args) == 2 else None
                    return eval_context["context"].get(key, default)
            raise ValueError("Unsupported function call")
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    def evaluate(self, liquid_score: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates rules against the score and context.
        Returns a dict with 'final_score', 'triggered_rules', 'verdict'.
        """
        final_score = liquid_score
        triggered = []
        verdict = "PROBABILISTIC" # Default

        for rule in self.rules:
            try:
                eval_context = {"liquid_score": liquid_score, "context": context}
                # Safe eval using AST module
                parsed_node = ast.parse(rule.condition_code, mode='eval').body
                if self._safe_eval_condition(parsed_node, eval_context):
                    triggered.append(rule.rule_id)
                    if rule.action == "FORCE_0":
                        final_score = 0.0
                        verdict = "ABORT"
                    elif rule.action == "FORCE_1":
                        final_score = 1.0
                        verdict = "EXECUTE"
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

        return {
            "final_score": final_score,
            "triggered_rules": triggered,
            "verdict": verdict
        }

class FrameCollapser:
    """
    Collapses the wave function (Probabilistic LNN output) into a Deterministic Plan.
    """
    def __init__(self):
        self.script_engine = DeterministicProbabilityScript()

    def collapse(self, plan: ExecutionPlan, liquid_state: Any, context: Dict[str, Any] = None) -> RefinedExecutionPlan:
        """
        Takes the raw plan and the liquid state (which implies a probability/score),
        applies constraints, and returns the Refined Plan.
        """
        if context is None:
            context = {}

        # Extract a 'score' from the liquid_state.
        # We assume the liquid_state vector's magnitude or mean represents the "Activation Energy".
        liquid_score = 0.5
        try:
            if hasattr(liquid_state, 'mean'):
                # Torch tensor
                liquid_score = float(liquid_state.mean())
            elif hasattr(liquid_state, 'data'):
                # MockTensor
                # In our MockTensor, data might not be populated or might be a list
                if isinstance(liquid_state.data, list) and len(liquid_state.data) > 0:
                     # Just take a dummy value or the first value if numeric
                     val = liquid_state.data[0]
                     if isinstance(val, (int, float)):
                         liquid_score = float(val)
                     else:
                         liquid_score = 0.5
            elif isinstance(liquid_state, (float, int)):
                liquid_score = float(liquid_state)
        except Exception as e:
            logger.warning(f"Could not extract score from liquid state: {e}. Defaulting to 0.5")
            liquid_score = 0.5

        # Clamp score 0-1 (if not already)
        liquid_score = max(0.0, min(1.0, abs(liquid_score)))

        evaluation = self.script_engine.evaluate(liquid_score, context)

        final_verdict = evaluation["verdict"]
        if final_verdict == "PROBABILISTIC":
            final_verdict = "EXECUTE" if evaluation["final_score"] > 0.5 else "HOLD"

        return RefinedExecutionPlan(
            **plan.model_dump(),
            collapser_verdict=final_verdict,
            original_probability=liquid_score,
            applied_constraints=evaluation["triggered_rules"]
        )
