import asyncio
import json
import uuid
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

class TokenOverflowException(Exception):
    pass

class TaskNode:
    def __init__(
        self,
        task_id: str,
        coroutine_func: Callable[..., Coroutine[Any, Any, Any]],
        dependencies: Optional[List[str]] = None,
        max_tokens: int = 1000,
        timeout: float = 60.0,
        is_human_gate: bool = False
    ):
        self.task_id = task_id
        self.coroutine_func = coroutine_func
        self.dependencies = dependencies or []
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.is_human_gate = is_human_gate

class StateLedger:
    def __init__(self, workflow_id: str, trace_id: Optional[str] = None):
        self.workflow_id = workflow_id
        self.trace_id = trace_id or f"tr-{uuid.uuid4().hex[:12]}-2026"
        self.status = "PENDING"
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.global_context: Dict[str, Any] = {}
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def log_event(self, event: str, task_id: str, span_id: str, metrics: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event,
            "task_id": task_id,
            "span_id": span_id
        }
        if metrics is not None:
            entry["metrics"] = metrics
        self.execution_history.append(entry)
        # Log telemetry
        print(json.dumps(entry))

    def update_task_status(self, task_id: str, status: str, span_id: str, parent_id: Optional[str] = None):
        self.task_registry[task_id] = {
            "status": status,
            "span_id": span_id,
            "parent_id": parent_id
        }

    def serialize(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "workflow_id": self.workflow_id,
                "trace_id": self.trace_id,
                "status": self.status,
                "start_time": self.start_time,
                "end_time": self.end_time
            },
            "global_context": self.global_context,
            "task_registry": self.task_registry,
            "execution_history": self.execution_history
        }

    def load(self, data: Dict[str, Any]):
        self.workflow_id = data["metadata"]["workflow_id"]
        self.trace_id = data["metadata"]["trace_id"]
        self.status = data["metadata"]["status"]
        self.start_time = data["metadata"]["start_time"]
        self.end_time = data["metadata"]["end_time"]
        self.global_context = data.get("global_context", {})
        self.task_registry = data.get("task_registry", {})
        self.execution_history = data.get("execution_history", [])

class OrchestratorEngine:
    def __init__(self, state_ledger: StateLedger):
        self.ledger = state_ledger
        self.tasks: Dict[str, TaskNode] = {}

    def add_task(self, task: TaskNode):
        self.tasks[task.task_id] = task
        if task.task_id not in self.ledger.task_registry:
            self.ledger.update_task_status(task.task_id, "PENDING", f"sp-{uuid.uuid4().hex[:8]}", None)

    def load_checkpoint(self, checkpoint_data: Dict[str, Any]):
        self.ledger.load(checkpoint_data)

    def _get_parent_span_id(self, task: TaskNode) -> Optional[str]:
        if not task.dependencies:
            return None
        primary_dep = task.dependencies[0]
        return self.ledger.task_registry.get(primary_dep, {}).get("span_id")

    async def execute_task(self, task: TaskNode, *args, **kwargs):
        registry_entry = self.ledger.task_registry.get(task.task_id, {})
        span_id = registry_entry.get("span_id") or f"sp-{uuid.uuid4().hex[:8]}"
        parent_id = self._get_parent_span_id(task)

        self.ledger.update_task_status(task.task_id, "RUNNING", span_id, parent_id)
        self.ledger.log_event("TASK_START", task.task_id, span_id)

        try:
            result = await asyncio.wait_for(
                task.coroutine_func(self.ledger.global_context, *args, **kwargs),
                timeout=task.timeout
            )

            if task.is_human_gate and result == "SUSPENDED_AWAITING_INPUT":
                self.ledger.update_task_status(task.task_id, "SUSPENDED", span_id, parent_id)
                self.ledger.log_event("TASK_SUSPENDED", task.task_id, span_id)
                self.ledger.status = "SUSPENDED"
                return "SUSPENDED"

            tokens_used = result.get("tokens_used", 0) if isinstance(result, dict) else 0

            if tokens_used > task.max_tokens:
                raise TokenOverflowException(f"Token overflow: {tokens_used} > {task.max_tokens}")

            self.ledger.update_task_status(task.task_id, "COMPLETED", span_id, parent_id)
            self.ledger.log_event("TASK_SUCCESS", task.task_id, span_id, metrics={"tokens_used": tokens_used})

            if isinstance(result, dict) and "updates" in result:
                self.ledger.global_context.update(result["updates"])

            return "COMPLETED"

        except TokenOverflowException as e:
            self.ledger.update_task_status(task.task_id, "FAILED", span_id, parent_id)
            self.ledger.log_event("TASK_FAILED", task.task_id, span_id, metrics={"error": str(e), "type": "TokenOverflowException"})
            raise
        except Exception as e:
            self.ledger.update_task_status(task.task_id, "FAILED", span_id, parent_id)
            self.ledger.log_event("TASK_FAILED", task.task_id, span_id, metrics={"error": str(e)})
            raise

    async def run(self):
        if not self.ledger.start_time:
            self.ledger.start_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.ledger.status != "SUSPENDED":
            self.ledger.status = "RUNNING"

        while True:
            ready_tasks = []
            for task_id, task in self.tasks.items():
                status = self.ledger.task_registry.get(task_id, {}).get("status", "PENDING")
                if status == "PENDING":
                    deps_statuses = [
                        self.ledger.task_registry.get(dep, {}).get("status")
                        for dep in task.dependencies
                    ]

                    if any(s in ("FAILED", "SKIPPED") for s in deps_statuses):
                        span_id = self.ledger.task_registry.get(task_id, {}).get("span_id") or f"sp-{uuid.uuid4().hex[:8]}"
                        parent_id = self._get_parent_span_id(task)
                        self.ledger.update_task_status(task_id, "SKIPPED", span_id, parent_id)
                        self.ledger.log_event("TASK_SKIPPED", task_id, span_id)
                    elif all(s == "COMPLETED" for s in deps_statuses):
                        ready_tasks.append(task)

            if not ready_tasks:
                break

            results = await asyncio.gather(*(self.execute_task(task) for task in ready_tasks), return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    # Continue the loop so SKIPPED statuses can propagate
                    pass
                elif result == "SUSPENDED":
                    self.ledger.end_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    return

        all_done = all(
            self.ledger.task_registry.get(t, {}).get("status") in ("COMPLETED", "SKIPPED")
            for t in self.tasks
        )

        if all_done:
            self.ledger.status = "COMPLETED"
        elif any(self.ledger.task_registry.get(t, {}).get("status") == "FAILED" for t in self.tasks):
             self.ledger.status = "FAILED"

        self.ledger.end_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def resolve_human_gate(self, task_id: str, callback_payload: Dict[str, Any]):
        if self.ledger.task_registry.get(task_id, {}).get("status") == "SUSPENDED":
            span_id = self.ledger.task_registry.get(task_id, {}).get("span_id")
            parent_id = self.ledger.task_registry.get(task_id, {}).get("parent_id")

            self.ledger.global_context.update(callback_payload.get("updates", {}))
            tokens_used = callback_payload.get("tokens_used", 0)

            self.ledger.update_task_status(task_id, "COMPLETED", span_id, parent_id)
            self.ledger.log_event("TASK_SUCCESS", task_id, span_id, metrics={"tokens_used": tokens_used, "event": "HUMAN_GATE_RESOLVED"})
            self.ledger.status = "PENDING"
