"""h_ctx — pre-turn hook: mask stale observations and enforce the context token budget."""
import json
import math
from typing import Any, Dict, Tuple

from core.engine.harness.policy import ContextPolicy, HarnessHalt
from core.schemas.agent_schema import AgentInput


def mask_stale_observations(context: Dict[str, Any], policy: ContextPolicy) -> Tuple[Dict[str, Any], int]:
    """Replace all but the newest ``keep_last_observations`` entries with a placeholder (returns a copy)."""
    observations = context.get(policy.observations_key)
    if not isinstance(observations, list) or len(observations) <= policy.keep_last_observations:
        return context, 0
    stale = len(observations) - policy.keep_last_observations
    masked = [policy.masked_placeholder] * stale + observations[stale:]
    return {**context, policy.observations_key: masked}, stale


def estimate_tokens(payload: Any, chars_per_token: int) -> int:
    try:
        text = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(payload)
    return math.ceil(len(text) / chars_per_token)


def pre_turn(input_data: AgentInput, policy: ContextPolicy) -> Tuple[AgentInput, Dict[str, Any]]:
    """Return the masked input and a report; raise ``HarnessHalt`` if the budget is still exceeded."""
    context, masked = mask_stale_observations(input_data.context, policy)
    tokens = estimate_tokens({"query": input_data.query, "context": context}, policy.chars_per_token)
    report = {"masked_observations": masked, "estimated_tokens": tokens}
    if tokens > policy.max_context_tokens:
        raise HarnessHalt("context_budget_exceeded", {**report, "max_context_tokens": policy.max_context_tokens})
    return input_data.model_copy(update={"context": context}), report
