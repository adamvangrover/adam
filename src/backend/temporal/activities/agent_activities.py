from typing import Dict, Any
from temporalio import activity
import structlog
from src.backend.orchestration.agent_runtime import AgentTask, AgentOrchestrator

logger = structlog.get_logger(__name__)

@activity.defn
async def execute_agent_task_activity(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Idempotent Temporal activity for executing a sub-agent task.
    """
    logger.info("activity_started", task_id=task_dict.get("task_id"))

    # Reconstruct Pydantic model
    task = AgentTask(**task_dict)

    # In a real environment, Qdrant client would be initialized here
    orchestrator = AgentOrchestrator(memory_client=None, rules_path="rules")

    # Execute deterministic logic
    result = await orchestrator.execute_task(task)

    logger.info("activity_completed", task_id=str(task.task_id), status=result.status)
    return result.model_dump(mode='json')
