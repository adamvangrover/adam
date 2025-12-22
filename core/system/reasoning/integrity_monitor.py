"""
Agent Notes:
    - Role: Principal AI Architect
    - Module: Reasoning Integrity Monitor
    - Purpose: Provides defensive grounding for agent reasoning chains.
      It validates financial constraints, logical consistency, and data existence
      before actions are executed or reports are finalized.
    - Philosophy: Financial systems cannot fail silently. Hallucinations in
      financial advice are unacceptable. This module acts as a 'Logic Gate'.
    - Future State: Will integrate with Z3 Theorem Prover for formal verification
      of complex financial contracts.
"""

import logging
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, ValidationError
import numpy as np

# Absolute imports as per standard
from core.utils.logging_utils import get_logger  # Assuming existence, falling back to standard if needed

# Configure Logging
logger = logging.getLogger("Adam_Integrity_Monitor")
logger.setLevel(logging.INFO)


class FinancialConstraint(BaseModel):
    """Defines a hard constraint for financial reasoning."""
    constraint_id: str
    description: str
    expression: str  # Python evaluable string for flexibility in v1
    tolerance: float = 0.001


class ValidationResult(BaseModel):
    """Standardized output for validation checks."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}


class IntegrityMonitor:
    """
    The IntegrityMonitor inspects reasoning chains and data payloads
    to ensure they adhere to financial logic and system constraints.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._setup_default_constraints()

    def _setup_default_constraints(self):
        """Initializes the core set of financial invariants."""
        self.constraints = {
            "probability_sum": FinancialConstraint(
                constraint_id="PROB_SUM_1",
                description="Sum of probabilities in a scenario analysis must equal 1.0",
                expression="abs(sum(values) - 1.0) < tolerance"
            ),
            "risk_score_range": FinancialConstraint(
                constraint_id="RISK_RANGE",
                description="Risk scores must be between 0 and 100",
                expression="all(0 <= v <= 100 for v in values)"
            ),
            "no_negative_valuation": FinancialConstraint(
                constraint_id="POS_VALUATION",
                description="Asset valuation cannot be negative unless explicitly short",
                expression="all(v >= 0 for v in values)"
            )
        }

    def validate_financial_metrics(self, metrics: Dict[str, float], context: str = "general") -> ValidationResult:
        """
        Validates a dictionary of financial metrics against known logical constraints.

        Args:
            metrics: Key-value pair of metric names and values.
            context: The context of validation (e.g., 'risk_report', 'portfolio_opt').
        """
        errors = []

        try:
            # Example Check 1: Probability Distributions
            prob_keys = [k for k in metrics.keys() if "probability" in k.lower()]
            if prob_keys:
                prob_values = [metrics[k] for k in prob_keys]
                # Only validate sum if it looks like a complete distribution (heuristic)
                if len(prob_values) > 1:
                    if not abs(sum(prob_values) - 1.0) < 0.01:
                        errors.append(f"Probabilities {prob_keys} sum to {sum(prob_values)}, expected 1.0")

            # Example Check 2: Margin Logic
            if "revenue" in metrics and "net_income" in metrics:
                if metrics["net_income"] > metrics["revenue"]:
                    errors.append(
                        f"Net Income ({metrics['net_income']}) exceeds Revenue ({metrics['revenue']}). Impossible.")

            # Example Check 3: Volatility
            if "volatility" in metrics and metrics["volatility"] < 0:
                errors.append(f"Volatility ({metrics['volatility']}) cannot be negative.")

        except Exception as e:
            logger.error(f"Error during metric validation: {str(e)}")
            errors.append(f"System error during validation: {str(e)}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_reasoning_graph(self, nodes: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validates a graph of reasoning steps for circular logic or disconnected nodes.
        Expected node format: {'id': str, 'dependencies': List[str], 'content': str}
        """
        errors = []
        visited = set()
        recursion_stack = set()

        def detect_cycle(node_id, graph):
            visited.add(node_id)
            recursion_stack.add(node_id)

            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if detect_cycle(neighbor, graph):
                        return True
                elif neighbor in recursion_stack:
                    return True

            recursion_stack.remove(node_id)
            return False

        # Build Adjacency List
        graph = {n['id']: n.get('dependencies', []) for n in nodes}

        # Check dependencies exist
        all_ids = set(graph.keys())
        for node_id, deps in graph.items():
            for dep in deps:
                if dep not in all_ids:
                    errors.append(f"Node {node_id} depends on non-existent node {dep}")

        # Check for cycles
        for node_id in graph:
            if node_id not in visited:
                if detect_cycle(node_id, graph):
                    errors.append(f"Circular reasoning detected starting at or involving {node_id}")
                    break

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def enforce_data_grounding(self, text: str, verified_sources: List[str]) -> ValidationResult:
        """
        Ensures that specific claims in the text are grounded in the verified sources.
        This is a heuristic check for hallucination prevention.
        """
        # Placeholder for advanced NLP grounding logic.
        # In v23, this would use embeddings to check similarity.
        # For v1, we check for data citation format.

        warnings = []
        if "According to" in text or "reported that" in text:
            found_source = any(source in text for source in verified_sources)
            if not found_source:
                warnings.append("Text contains citation phrasing but no verified source path was matched.")

        return ValidationResult(is_valid=True, warnings=warnings)


if __name__ == "__main__":
    # Quick self-test
    monitor = IntegrityMonitor()

    # Test Logic
    test_metrics = {"revenue": 100.0, "net_income": 150.0}
    result = monitor.validate_financial_metrics(test_metrics)
    print(f"Test 1 (Should Fail): {result.is_valid} - {result.errors}")

    test_metrics_2 = {"scenario_A_probability": 0.4, "scenario_B_probability": 0.6}
    result_2 = monitor.validate_financial_metrics(test_metrics_2)
    print(f"Test 2 (Should Pass): {result_2.is_valid} - {result_2.errors}")
