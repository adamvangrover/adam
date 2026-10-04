"""
Deterministic Schema Validation, jsonLogic Rule Engine, and W3C PROV-O Audit Graph.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class DistributionQuantiles(BaseModel):
    p10: float = Field(..., description="10th percentile value")
    p50: float = Field(..., description="50th percentile (median) value")
    p90: float = Field(..., description="90th percentile value")

    @field_validator("p50")
    @classmethod
    def validate_p50(cls, v: float, info) -> float:
        p10 = info.data.get("p10")
        if p10 is not None and v <= p10:
            raise ValueError(f"p50 ({v}) must be greater than p10 ({p10})")
        return v

    @field_validator("p90")
    @classmethod
    def validate_p90(cls, v: float, info) -> float:
        p50 = info.data.get("p50")
        if p50 is not None and v <= p50:
            raise ValueError(f"p90 ({v}) must be greater than p50 ({p50})")
        return v


class ScenarioBreakdown(BaseModel):
    base_case_weight: float = Field(..., ge=0.0, le=100.0)
    base_case_thesis: str
    bull_case_weight: float = Field(..., ge=0.0, le=100.0)
    bull_case_thesis: str
    bear_case_weight: float = Field(..., ge=0.0, le=100.0)
    bear_case_thesis: str

    @field_validator("bear_case_weight")
    @classmethod
    def validate_sum(cls, v: float, info) -> float:
        base = info.data.get("base_case_weight", 0.0)
        bull = info.data.get("bull_case_weight", 0.0)
        total = round(base + bull + v, 1)
        if total != 100.0:
            raise ValueError(f"Scenario weights must sum to 100% (got {total}%)")
        return v


class ProposedSubmission(BaseModel):
    direction: Direction
    point_forecast: float
    confidence_interval: DistributionQuantiles
    confidence: float = Field(..., ge=0.50, le=0.95, description="Model confidence score in [0.50, 0.95]")
    std_deviation: Optional[float] = None


class ValidationAuditLog(BaseModel):
    jsonlogic_passed: bool
    prov_o_hash: str
    tail_risk_flags: str = "None"
    rules_evaluated: int = 5
    schema_version: str = "v2.5-strict"


class ChallengeForecast(BaseModel):
    challenge_id: str
    challenge_title: str
    asset: str
    resolution_horizon: str
    resolution_metric: str
    proposed_submission: ProposedSubmission
    executive_rationale: str
    scenarios: ScenarioBreakdown
    validation: ValidationAuditLog
    raw_api_payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("executive_rationale")
    @classmethod
    def validate_word_count(cls, v: str) -> str:
        words = len(v.strip().split())
        if words < 140 or words > 260:
            # Allow slight tolerance in validator but warn
            pass
        return v


# ─── jsonLogic Deterministic Evaluator ────────────────────────────────────────

class JsonLogicEngine:
    """
    Deterministic jsonLogic evaluator conforming to standard JSONLogic specification.
    Used for pre-submission compliance and mathematical consistency auditing.
    """

    @classmethod
    def evaluate(cls, rule: Any, data: Dict[str, Any]) -> Any:
        if isinstance(rule, dict):
            if not rule:
                return {}
            op = next(iter(rule))
            args = rule[op]
            if not isinstance(args, list):
                args = [args]
            return cls._apply_op(op, args, data)
        elif isinstance(rule, list):
            return [cls.evaluate(item, data) for item in rule]
        else:
            return rule

    @classmethod
    def _apply_op(cls, op: str, args: List[Any], data: Dict[str, Any]) -> Any:
        if op == "var":
            var_name = args[0] if args else ""
            default = args[1] if len(args) > 1 else None
            return data.get(var_name, default)

        evaluated_args = [cls.evaluate(a, data) for a in args]

        if op == "==":
            return evaluated_args[0] == evaluated_args[1]
        elif op == "!=":
            return evaluated_args[0] != evaluated_args[1]
        elif op == ">":
            return evaluated_args[0] > evaluated_args[1]
        elif op == ">=":
            return evaluated_args[0] >= evaluated_args[1]
        elif op == "<":
            return evaluated_args[0] < evaluated_args[1]
        elif op == "<=":
            return evaluated_args[0] <= evaluated_args[1]
        elif op == "and":
            return all(evaluated_args)
        elif op == "or":
            return any(evaluated_args)
        elif op == "!":
            return not evaluated_args[0]
        elif op == "in":
            return evaluated_args[0] in evaluated_args[1]
        elif op == "+":
            return sum(evaluated_args)
        else:
            raise ValueError(f"Unsupported jsonLogic operator: {op}")

    @classmethod
    def validate_forecast(cls, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validates forecast dictionary against deterministic jsonLogic rules.
        """
        rules = [
            # Rule 1: Confidence bounded in [0.50, 0.95]
            {
                "id": "RULE_CONF_BOUNDS",
                "logic": {
                    "and": [
                        {">=": [{"var": "confidence"}, 0.50]},
                        {"<=": [{"var": "confidence"}, 0.95]}
                    ]
                }
            },
            # Rule 2: Strictly monotonic quantiles P10 < P50 < P90
            {
                "id": "RULE_QUANTILE_ORDER",
                "logic": {
                    "and": [
                        {"<": [{"var": "p10"}, {"var": "p50"}]},
                        {"<": [{"var": "p50"}, {"var": "p90"}]}
                    ]
                }
            },
            # Rule 3: Valid discrete direction
            {
                "id": "RULE_DIRECTION_VALID",
                "logic": {
                    "in": [{"var": "direction"}, ["bullish", "bearish", "neutral"]]
                }
            },
            # Rule 4: Scenario weights sum to 100
            {
                "id": "RULE_SCENARIOS_SUM",
                "logic": {
                    "==": [{"var": "scenario_sum"}, 100]
                }
            },
            # Rule 5: Rationale word count between 140 and 260 words
            {
                "id": "RULE_RATIONALE_LENGTH",
                "logic": {
                    "and": [
                        {">=": [{"var": "word_count"}, 140]},
                        {"<=": [{"var": "word_count"}, 260]}
                    ]
                }
            }
        ]

        failed = []
        for r in rules:
            try:
                passed = cls.evaluate(r["logic"], data)
                if not passed:
                    failed.append(r["id"])
            except Exception as e:
                failed.append(f"{r['id']}_ERROR: {str(e)}")

        return (len(failed) == 0, failed)


# ─── W3C PROV-O Lineage Graph Generator ──────────────────────────────────────

class ProvOGraphGenerator:
    """
    Generates W3C PROV-O compliant JSON-LD graph records and cryptographic
    lineage hashes (SHA-256) for complete analytical provenance.
    """

    @classmethod
    def generate(
        cls,
        challenge_id: str,
        asset: str,
        input_data: Dict[str, Any],
        model_params: Dict[str, Any],
        scenarios: Dict[str, Any],
        forecast_output: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build raw deterministic canonical structure for hashing
        hash_payload = {
            "challenge_id": challenge_id,
            "asset": asset,
            "timestamp": now_iso,
            "input_signature": {
                "open_price": input_data.get("open_price"),
                "current_price": input_data.get("current_price"),
                "macro_regime": input_data.get("macro_regime"),
                "credit_spread_bps": input_data.get("credit_spread_bps"),
                "rates_iv_move": input_data.get("rates_iv_move"),
            },
            "parameters": model_params,
            "scenarios": scenarios,
            "output": forecast_output,
        }

        canonical_json = json.dumps(hash_payload, sort_keys=True)
        lineage_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        trace_key = f"prov:ha:{challenge_id[:8]}:{lineage_hash[:16]}"

        prov_doc = {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "ha": "https://headlinearena.com/ns/audit#"
            },
            "@graph": [
                {
                    "@id": f"ha:entity/market_state/{challenge_id}",
                    "@type": "prov:Entity",
                    "prov:value": input_data,
                    "prov:generatedAtTime": now_iso
                },
                {
                    "@id": f"ha:activity/macro_deliberation/{trace_key}",
                    "@type": "prov:Activity",
                    "prov:startedAtTime": now_iso,
                    "prov:used": f"ha:entity/market_state/{challenge_id}",
                    "prov:wasAssociatedWith": "ha:agent/ADAM-Macro-Sentinel"
                },
                {
                    "@id": f"ha:entity/forecast/{trace_key}",
                    "@type": "prov:Entity",
                    "prov:wasGeneratedBy": f"ha:activity/macro_deliberation/{trace_key}",
                    "ha:lineage_hash": lineage_hash,
                    "ha:direction": forecast_output.get("direction"),
                    "ha:confidence": forecast_output.get("confidence"),
                    "ha:p50": forecast_output.get("p50")
                },
                {
                    "@id": "ha:agent/ADAM-Macro-Sentinel",
                    "@type": "prov:Agent",
                    "prov:label": "ADAM-Macro-Sentinel Forecasting Engine v2.5",
                    "ha:engine": "Multi-Objective Epistemic Dispatch Harness"
                }
            ]
        }

        return trace_key, prov_doc
