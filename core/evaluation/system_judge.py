from typing import Dict, Any, List, Optional
import time
from pydantic import BaseModel, Field

class EvaluationRubric(BaseModel):
    criteria: str = Field(..., description="The main criteria being judged (e.g., Accuracy, Coherence, Formatting)")
    max_score: int = Field(5, description="Maximum possible score for this criteria")
    weight: float = Field(1.0, description="Weight of this criteria in the overall evaluation")
    description: str = Field(..., description="Detailed description of what constitutes a perfect score")

class SystemJudgeMetrics(BaseModel):
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics (prompt, completion, total)")
    token_efficiency: float = Field(..., description="Tokens generated per millisecond")
    format_valid: bool = Field(..., description="Whether the output strictly adhered to the required schema")

class SystemJudge:
    """
    Programmatic evaluator for deterministic checks: latency, token efficiency, formatting constraints.
    """
    def __init__(self):
        pass

    def evaluate(self, execution_stats: Dict[str, Any], output_text: str, expected_schema: Optional[Any] = None) -> SystemJudgeMetrics:
        """
        Evaluates the system-level metrics of an LLM call.
        """
        start_time = execution_stats.get('start_time', time.time())
        end_time = execution_stats.get('end_time', time.time())
        latency = (end_time - start_time) * 1000

        token_usage = execution_stats.get('token_usage', {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0})
        completion_tokens = token_usage.get('completion_tokens', len(output_text) // 4) # rough estimate if not provided

        efficiency = completion_tokens / latency if latency > 0 else 0.0

        format_valid = True
        if expected_schema:
            try:
                # Basic simulated schema check
                import json
                parsed = json.loads(output_text)
                if hasattr(expected_schema, 'model_validate'):
                    expected_schema.model_validate(parsed)
            except Exception:
                format_valid = False

        return SystemJudgeMetrics(
            latency_ms=latency,
            token_usage=token_usage,
            token_efficiency=efficiency,
            format_valid=format_valid
        )
