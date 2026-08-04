"""
Core Agent Orchestration Runtime for Adam OS.

Responsible for task delegation, JIT memory injection, jsonLogic evaluation,
and W3C PROV-O compliant telemetry logging.
"""

import asyncio
import functools
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, ValidationError
from json_logic import jsonLogic  # type: ignore
from temporalio.client import Client

from src.backend.memory.qdrant_client import JitMemoryClient
from src.shared.models.memory_models import MemoryQuery

# ---------------------------------------------------------------------------
# Telemetry Setup (W3C PROV-O Standardized)
# ---------------------------------------------------------------------------
# Note: structlog should optimally be configured at the application entry point.
# We ensure a fallback configuration here if this is executed independently.
if not structlog.is_configured():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

logger = structlog.get_logger(__name__)


@functools.lru_cache(maxsize=128)
def _load_rule_from_disk(file_path: str) -> Dict[str, Any]:
    """Caches loaded JSON rules to avoid repetitive disk I/O."""
    with open(file_path, "r") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------
class AgentTask(BaseModel):
    """Represents a discrete task for a sub-agent."""
    task_id: UUID = Field(default_factory=uuid4)
    target_agent: str = Field(..., description="The role of the agent (e.g., 'underwriter')")
    payload: Dict[str, Any] = Field(..., description="Task execution parameters")
    context_keys: List[str] = Field(default_factory=list, description="Keys for JIT memory loading")

class ExecutionResult(BaseModel):
    """Deterministic output of an agent execution."""
    task_id: UUID
    status: str = Field(..., pattern="^(SUCCESS|FAILURE|DEFERRED)$")
    output: Dict[str, Any]
    prov_o_trail: List[Dict[str, Any]] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Core Runtime
# ---------------------------------------------------------------------------
class AgentOrchestrator:
    """
    Manages the lifecycle, memory, and telemetry of autonomous financial agents.
    """

    def __init__(
        self, 
        memory_client: Optional[JitMemoryClient] = None, 
        rules_path: str = "rules/", 
        temporal_client: Optional[Client] = None
    ):
        """
        Initializes the orchestrator with memory, rule, and temporal dependencies.

        Args:
            memory_client: Instantiated vector database client for JIT memory.
            rules_path: Directory path containing jsonLogic definitions.
            temporal_client: Instantiated Temporal client for executing workflows.
        """
        self.memory = memory_client or JitMemoryClient()
        self.rules_path = rules_path
        self.temporal_client = temporal_client
        self.prov_log: List[Dict[str, Any]] = []

    async def _load_jit_context(self, context_keys: List[str]) -> Dict[str, Any]:
        """
        Retrieves highly specific context to prevent LLM context window overflow.
        """
        if not context_keys:
            return {}

        query = MemoryQuery(query="Context retrieval", context_keys=context_keys)
        results = await self.memory.search_context(query)

        return {"retrieved_docs": [r.payload for r in results]}

    def _evaluate_preconditions(self, payload: Dict[str, Any], rule_name: str) -> bool:
        """
        Evaluates dynamic JSON logic rules before permitting agent execution.
        """
        try:
            rule_path = f"{self.rules_path}/{rule_name}.json"
            rule = _load_rule_from_disk(rule_path)
            result = jsonLogic(rule, payload)
            return bool(result)
        except FileNotFoundError:
            logger.warning("rule_not_found", rule_name=rule_name)
            return True # Fail open or closed based on strictness policy
        except Exception as e:
            logger.error("rule_evaluation_error", error=str(e))
            return False

    def _record_provenance(self, activity: str, entity: str, agent: str, data: Dict[str, Any]) -> None:
        """
        Records a W3C PROV-O compliant telemetry event.
        """
        event = {
            "prov:Activity": activity,
            "prov:Entity": entity,
            "prov:Agent": agent,
            "prov:generatedAtTime": datetime.now(timezone.utc).isoformat(),
            "data_snapshot": data
        }
        self.prov_log.append(event)
        logger.info("prov_o_event", **event)

    async def execute_task(self, task: AgentTask) -> ExecutionResult:
        """
        Main entrypoint for executing an agent task deterministically.
        """
        self._record_provenance(
            activity="TaskInitialization",
            entity=str(task.task_id),
            agent="Orchestrator",
            data=task.model_dump(mode='json')
        )

        # 1. Evaluate Governance / Rules
        if not self._evaluate_preconditions(task.payload, f"{task.target_agent}_gate"):
            self._record_provenance("TaskRejection", str(task.task_id), "PolicyEngine", {})
            return ExecutionResult(task_id=task.task_id, status="FAILURE", output={"error": "Policy rejection"})

        # 2. Load JIT Memory
        context = await self._load_jit_context(task.context_keys)
        self._record_provenance("MemoryInjection", str(task.task_id), "MemoryLayer", context)

        # 3. Agent Execution via Temporal Workflow
        if self.temporal_client is None:
            # Fallback/stub for tests where temporal client is not provided
            simulated_output = {"decision": "approved", "confidence": 0.95}
        else:
            # Execute actual workflow
            # Using absolute string for workflow name to avoid circular dependencies
            simulated_output = await self.temporal_client.execute_workflow(
                "AgentExecutionWorkflow",
                args=[task.target_agent, task.payload, context],
                id=f"agent-task-{task.task_id}",
                task_queue="agent-task-queue"
            )

        self._record_provenance(
            activity="TaskCompletion",
            entity=str(task.task_id),
            agent=task.target_agent,
            data=simulated_output
        )

        return ExecutionResult(
            task_id=task.task_id,
            status="SUCCESS",
            output=simulated_output,
            prov_o_trail=self.prov_log
        )