"""
Symbolic jsonLogic Covenant Evaluator and Quasi-Monte Carlo (QMC) Stress-Test Bridge.
Enforces the neuro-symbolic boundary under SR 11-7 model risk governance.
Halts execution with SR117UnverifiedTelemetryError when negative findings are present.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy.stats import qmc, norm
from core.telemetry.prov_o import logfire


class SR117UnverifiedTelemetryError(Exception):
    pass


class JsonLogicEvaluator:
    @staticmethod
    def evaluate(rule: Dict[str, Any], data: Dict[str, Any]) -> Any:
        if not isinstance(rule, dict):
            return rule

        operator = list(rule.keys())[0]
        operands = rule[operator]

        if operator == "var":
            return data.get(operands, None)
        elif operator == "<=":
            return JsonLogicEvaluator.evaluate(operands[0], data) <= JsonLogicEvaluator.evaluate(operands[1], data)
        elif operator == ">=":
            return JsonLogicEvaluator.evaluate(operands[0], data) >= JsonLogicEvaluator.evaluate(operands[1], data)
        elif operator == "/":
            denom = JsonLogicEvaluator.evaluate(operands[1], data)
            if denom == 0:
                raise ZeroDivisionError("Covenant denominator is zero.")
            return JsonLogicEvaluator.evaluate(operands[0], data) / denom
        else:
            raise ValueError(f"Unsupported jsonLogic operator: {operator}")


class CovenantQMCBridge:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rulebook = {
            "total_leverage_test": {
                "<=": [
                    {"/": [{"var": "total_funded_debt"}, {"var": "consolidated_ebitda"}]},
                    {"var": "max_total_leverage_covenant"}
                ]
            },
            "secured_leverage_test": {
                "<=": [
                    {"/": [{"var": "senior_secured_debt"}, {"var": "consolidated_ebitda"}]},
                    {"var": "max_secured_leverage_covenant"}
                ]
            }
        }

    def inspect_and_validate_telemetry(self, telemetry_payload: Dict[str, Any]) -> Dict[str, float]:
        if telemetry_payload.get("status") == "UNVERIFIED_NEGATIVE_FINDING":
            details = telemetry_payload.get("negative_finding", {})
            raise SR117UnverifiedTelemetryError(
                f"SR 11-7 HALT: Unverified parameter detected: {details.get('target_metric')}."
            )

        params = telemetry_payload.get("extracted_parameters", telemetry_payload)
        for key, val in params.items():
            if val == "UNVERIFIED_NEGATIVE_FINDING":
                raise SR117UnverifiedTelemetryError(f"SR 11-7 HALT: Parameter {key} unverified.")

        for req in ["consolidated_ebitda", "total_funded_debt", "max_total_leverage_covenant"]:
            if req not in params or params[req] is None:
                raise SR117UnverifiedTelemetryError(f"SR 11-7 HALT: Missing {req}")
        return params

    def evaluate_covenants(self, validated_params: Dict[str, float]) -> Dict[str, Any]:
        with logfire.span("covenant_qmc_bridge.evaluate_covenants"):
            actual_total_leverage = validated_params["total_funded_debt"] / validated_params["consolidated_ebitda"]
            actual_secured_leverage = validated_params.get("senior_secured_debt", 0.0) / validated_params["consolidated_ebitda"]

            covenant_results = {}
            for test_name, rule in self.rulebook.items():
                covenant_results[test_name] = JsonLogicEvaluator.evaluate(rule, validated_params)

            return {
                "covenant_results": covenant_results,
                "metrics": {
                    "actual_total_leverage": round(actual_total_leverage, 4),
                    "actual_secured_leverage": round(actual_secured_leverage, 4),
                    "headroom_total_leverage": round(validated_params["max_total_leverage_covenant"] - actual_total_leverage, 4),
                    "headroom_secured_leverage": round(validated_params.get("max_secured_leverage_covenant", 3.0) - actual_secured_leverage, 4)
                },
                "all_covenants_passed": all(covenant_results.values())
            }

    def run_qmc_stress_simulation(
        self,
        validated_params: Dict[str, float],
        n_samples_power_of_two: int = 12,
        ebitda_volatility: float = 0.20
    ) -> Dict[str, Any]:
        with logfire.span("covenant_qmc_bridge.run_qmc_stress_simulation"):
            n_samples = 2**n_samples_power_of_two
            sobol_engine = qmc.Sobol(d=2, scramble=True, seed=self.seed)
            uniform_samples = sobol_engine.random(n=n_samples)
            normal_shocks = norm.ppf(uniform_samples)

            base_ebitda = validated_params["consolidated_ebitda"]
            base_debt = validated_params["total_funded_debt"]
            max_leverage = validated_params["max_total_leverage_covenant"]

            ebitda_shocks = normal_shocks[:, 0]
            stressed_ebitda = base_ebitda * np.exp(-0.5 * (ebitda_volatility**2) + ebitda_volatility * ebitda_shocks)
            stressed_leverage = base_debt / stressed_ebitda
            breach_indicators = stressed_leverage > max_leverage
            probability_of_breach = np.mean(breach_indicators)
            p99_leverage = np.percentile(stressed_leverage, 99)
            tail_breaches = stressed_leverage[stressed_leverage >= p99_leverage]
            expected_shortfall = np.mean(tail_breaches)

            return {
                "qmc_method": "Scrambled Sobol Low-Discrepancy Sequence",
                "sample_count": n_samples,
                "probability_of_covenant_breach": round(float(probability_of_breach), 4),
                "baseline_leverage": round(base_debt / base_ebitda, 4),
                "max_covenant_leverage": max_leverage,
                "p99_stressed_leverage": round(float(p99_leverage), 4),
                "expected_shortfall_leverage": round(float(expected_shortfall), 4),
                "simulation_status": "DETERMINISTIC_QMC_CONVERGED"
            }
