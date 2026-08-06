from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime, timezone

import structlog
from temporalio import activity

from src.backend.orchestration.agent_runtime import AgentTask, AgentOrchestrator

logger = structlog.get_logger(__name__)


@dataclass
class AgentExecutionParams:
    """Parameters for isolated agent inference."""
    agent_name: str
    payload: Dict[str, Any]
    context: Dict[str, Any]


@activity.defn
async def execute_agent_task_activity(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Idempotent Temporal activity for executing an entire sub-agent task orchestration.
    This encapsulates the initialization, memory retrieval, and policy evaluation.
    """
    logger.info("activity_started", activity="execute_agent_task", task_id=task_dict.get("task_id"))

    # Reconstruct Pydantic model
    task = AgentTask(**task_dict)

    # In a real environment, Qdrant client would be initialized here
    orchestrator = AgentOrchestrator(memory_client=None, rules_path="rules")

    # Execute deterministic logic
    result = await orchestrator.execute_task(task)

    logger.info("activity_completed", activity="execute_agent_task", task_id=str(task.task_id), status=result.status)
    return result.model_dump(mode='json')


@activity.defn
async def execute_agent_inference(params: AgentExecutionParams) -> Dict[str, Any]:
    """
    Idempotent activity to execute an agent's LLM inference or calculation.
    All non-deterministic side-effects must occur here.
    """
    logger.info("activity_started", activity="execute_agent_inference", agent=params.agent_name)

    # In a real system, this calls an LLM or a Rust pricing engine.
    # For now, it returns a deterministic stub.
    simulated_output = {
        "decision": "approved",
        "confidence": 0.92,
        "processed_at": datetime.now(timezone.utc).isoformat()
    }

    return simulated_output