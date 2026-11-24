"""h_ctrl — pre-turn hook: turn cap and duplicate tool-call breaker, plus the bounded handoff loop."""
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from core.engine.harness.policy import ControlPolicy, HarnessHalt
from core.schemas.agent_schema import AgentOutput

Dispatch = Callable[[str, Dict[str, Any]], Awaitable[Any]]


@dataclass
class TurnState:
    turns: int = 0
    tool_calls: Counter = field(default_factory=Counter)


def pre_turn(state: TurnState, policy: ControlPolicy) -> None:
    """Count a turn; raise ``HarnessHalt`` once ``max_turns`` have already run."""
    if state.turns >= policy.max_turns:
        raise HarnessHalt("turn_cap_exceeded", {"turns": state.turns, "max_turns": policy.max_turns})
    state.turns += 1


def record_tool_call(state: TurnState, name: str, arguments: Any, policy: ControlPolicy) -> None:
    """Raise ``HarnessHalt`` when the same (name, arguments) call repeats more than allowed."""
    signature = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
    state.tool_calls[signature] += 1
    if state.tool_calls[signature] > policy.max_duplicate_tool_calls:
        raise HarnessHalt("duplicate_tool_call", {"tool": name, "count": state.tool_calls[signature],
                                                  "max_duplicate_tool_calls": policy.max_duplicate_tool_calls})


def pending_handoff(result: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(next_step, metadata)`` if the result requests a follow-up that may run unattended."""
    metadata = result.metadata if isinstance(result, AgentOutput) else (
        result.get("metadata") if isinstance(result, dict) else None)
    if not isinstance(metadata, dict) or not metadata.get("next_step"):
        return None
    if metadata.get("hitl_required") or metadata.get("confidence_tier") == "hard_abort":
        return None
    return str(metadata["next_step"]), metadata


async def follow_handoffs(result: Any, dispatch: Dispatch, state: TurnState, policy: ControlPolicy) -> Any:
    """Dispatch ``metadata.next_step`` requests through the orchestrator until none remain.

    ``dispatch`` returns the next result, or ``None`` when the step is not routable (the loop stops and the
    request stays visible in the last result). Each hop is a turn and a tool call, so loops are bounded.
    """
    while (handoff := pending_handoff(result)) is not None:
        step, metadata = handoff
        pre_turn(state, policy)
        record_tool_call(state, f"handoff:{step}",
                         {"query": metadata.get("query"), "payload": metadata.get("payload")}, policy)
        nxt = await dispatch(step, metadata)
        if nxt is None:
            break
        result = nxt
    return result
