import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone

class TaskState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SUSPENDED_AWAITING_INPUT = "SUSPENDED_AWAITING_INPUT"
    SKIPPED = "SKIPPED"

@dataclass
class TaskNode:
    id: str
    func: Callable[..., Coroutine[Any, Any, Any]]
    dependencies: List[str] = field(default_factory=list)
    max_tokens: int = 1000
    timeout: float = 30.0
    conditional_router: Optional[Callable[[Dict[str, Any]], bool]] = None
    is_human_gate: bool = False

class StateLedger:
    def __init__(self, workflow_id: str, trace_id: str):
        self.metadata = {
            "workflow_id": workflow_id,
            "trace_id": trace_id,
            "status": "RUNNING",
            "start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end_time": None
        }
        self.global_context: Dict[str, Any] = {}
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def log_event(self, event: str, task_id: str, span_id: str, metrics: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event,
            "task_id": task_id,
            "span_id": span_id
        }
        if metrics is not None:
            entry["metrics"] = metrics
        self.execution_history.append(entry)
        print(json.dumps(entry))

    def to_json(self) -> str:
        return json.dumps({
            "metadata": self.metadata,
            "global_context": self.global_context,
            "task_registry": self.task_registry,
            "execution_history": self.execution_history
        }, indent=2)

class SequenceEngine:
    def __init__(self, ledger: StateLedger):
        self.ledger = ledger

    async def execute_task(self, task: TaskNode, span_id: str, parent_id: Optional[str] = None):
        self.ledger.task_registry[task.id]["status"] = TaskState.RUNNING
        self.ledger.task_registry[task.id]["span_id"] = span_id
        self.ledger.task_registry[task.id]["parent_id"] = parent_id
        self.ledger.log_event("TASK_START", task.id, span_id)

        try:
            # Check dependencies dynamically
            for dep in task.dependencies:
                dep_status = self.ledger.task_registry.get(dep, {}).get("status")
                if dep_status != TaskState.COMPLETED and dep_status != TaskState.SKIPPED:
                    self.ledger.task_registry[task.id]["status"] = TaskState.SKIPPED
                    self.ledger.log_event("TASK_SKIPPED", task.id, span_id, {"reason": f"Dependency {dep} failed"})
                    return
                # If dependency was skipped, this should be skipped too
                if dep_status == TaskState.SKIPPED:
                    self.ledger.task_registry[task.id]["status"] = TaskState.SKIPPED
                    self.ledger.log_event("TASK_SKIPPED", task.id, span_id, {"reason": f"Dependency {dep} was skipped"})
                    return

            if task.conditional_router:
                if not task.conditional_router(self.ledger.global_context):
                    self.ledger.task_registry[task.id]["status"] = TaskState.SKIPPED
                    self.ledger.log_event("TASK_SKIPPED", task.id, span_id, {"reason": "Conditional routing evaluated to false"})
                    return

            if task.is_human_gate:
                self.ledger.task_registry[task.id]["status"] = TaskState.SUSPENDED_AWAITING_INPUT
                self.ledger.log_event("TASK_SUSPENDED", task.id, span_id, {"reason": "Awaiting human input"})
                return

            result, metrics = await asyncio.wait_for(task.func(self.ledger.global_context), timeout=task.timeout)

            self.ledger.task_registry[task.id]["status"] = TaskState.COMPLETED
            self.ledger.log_event("TASK_SUCCESS", task.id, span_id, metrics)
            return result

        except asyncio.TimeoutError:
            self.ledger.task_registry[task.id]["status"] = TaskState.FAILED
            self.ledger.log_event("TASK_FAILED", task.id, span_id, {"reason": "Timeout"})
        except Exception as e:
            self.ledger.task_registry[task.id]["status"] = TaskState.FAILED
            self.ledger.log_event("TASK_FAILED", task.id, span_id, {"reason": str(e)})

class OrchestrationEngine:
    def __init__(self, workflow_id: str, trace_id: str):
        self.ledger = StateLedger(workflow_id, trace_id)
        self.tasks: Dict[str, TaskNode] = {}
        self.sequence_engine = SequenceEngine(self.ledger)

    def register_task(self, task: TaskNode):
        self.tasks[task.id] = task
        if task.id not in self.ledger.task_registry:
            self.ledger.task_registry[task.id] = {"status": TaskState.PENDING, "span_id": f"sp-{task.id}", "parent_id": None}
            if task.dependencies:
                 self.ledger.task_registry[task.id]["parent_id"] = f"sp-{task.dependencies[0]}"

    def receive_human_input(self, task_id: str, context_updates: Dict[str, Any]):
        """Callback mechanism for human gate."""
        if task_id in self.tasks and self.ledger.task_registry[task_id]["status"] == TaskState.SUSPENDED_AWAITING_INPUT:
            self.ledger.global_context.update(context_updates)
            span_id = self.ledger.task_registry[task_id].get("span_id", f"sp-{task_id}")
            self.ledger.task_registry[task_id]["status"] = TaskState.COMPLETED
            self.ledger.log_event("TASK_SUCCESS", task_id, span_id, {"input": "Human intervention applied"})

    async def run(self):
        pending_tasks = set(self.tasks.keys())

        # Checkpointer Resume Logic
        # Copy pending_tasks to iterate over it safely
        for tid in list(pending_tasks):
            status = self.ledger.task_registry.get(tid, {}).get("status", TaskState.PENDING)
            if status == TaskState.COMPLETED or status == TaskState.SUSPENDED_AWAITING_INPUT:
                pending_tasks.discard(tid)
            elif status == TaskState.SKIPPED:
                 # Check if the skip was because of conditional router, or failed dependency
                 # We reset to PENDING to let it re-evaluate
                 self.ledger.task_registry[tid]["status"] = TaskState.PENDING
            else:
                self.ledger.task_registry[tid]["status"] = TaskState.PENDING

        while pending_tasks:
            ready_tasks = []
            for tid in pending_tasks:
                task = self.tasks[tid]
                deps_met = True
                for dep in task.dependencies:
                    dep_status = self.ledger.task_registry.get(dep, {}).get("status")
                    if dep_status not in [TaskState.COMPLETED, TaskState.SKIPPED, TaskState.FAILED]:
                        deps_met = False
                        break
                if deps_met:
                    ready_tasks.append(task)

            if not ready_tasks:
                # Deadlock or suspended
                break

            coroutines = []
            for task in ready_tasks:
                span_id = self.ledger.task_registry[task.id].get("span_id", f"sp-{task.id}")
                parent_id = self.ledger.task_registry[task.id].get("parent_id", None)
                coroutines.append(self.sequence_engine.execute_task(task, span_id, parent_id))

            await asyncio.gather(*coroutines)

            for task in ready_tasks:
                pending_tasks.discard(task.id)

            if any(self.ledger.task_registry[t.id]["status"] == TaskState.SUSPENDED_AWAITING_INPUT for t in ready_tasks):
                break

            # Discard implicitly skipped tasks that skipped themselves during execution
            for tid in list(pending_tasks):
                if self.ledger.task_registry.get(tid, {}).get("status") == TaskState.SKIPPED:
                    pending_tasks.discard(tid)

        self.ledger.metadata["end_time"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.ledger.metadata["status"] = "COMPLETED" if all(v["status"] in [TaskState.COMPLETED, TaskState.SKIPPED] for v in self.ledger.task_registry.values()) else "INCOMPLETE"

        return self.ledger
