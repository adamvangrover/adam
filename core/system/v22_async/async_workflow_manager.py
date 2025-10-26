# core/system/v22_async/async_workflow_manager.py
import asyncio
import json
import uuid
from threading import RLock
from typing import Dict, Any, Optional

from core.system.message_broker import MessageBroker
from .workflow import AsyncWorkflow

class AsyncWorkflowManager:
    _instance: Optional['AsyncWorkflowManager'] = None
    _lock = RLock()

    def __init__(self):
        self.message_broker = MessageBroker.get_instance()

    @classmethod
    def get_instance(cls) -> 'AsyncWorkflowManager':
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    async def execute_workflow(self, workflow: AsyncWorkflow) -> Dict[str, Any]:
        """
        Executes a workflow asynchronously using the message broker.
        """
        results: Dict[str, Any] = {}
        futures: Dict[str, asyncio.Future] = {name: asyncio.Future() for name in workflow.tasks}
        remaining_deps = {name: set(task.dependencies) for name, task in workflow.tasks.items()}
        lock = RLock()

        def _on_task_completed(task_name: str, result: Any):
            with lock:
                if not futures[task_name].done():
                    futures[task_name].set_result(result)
                    results[task_name] = result

                for name, deps in remaining_deps.items():
                    if task_name in deps:
                        deps.remove(task_name)

                asyncio.create_task(_schedule_tasks())

        def _message_handler(task_name: str):
            def handle(message):
                data = json.loads(message)
                result = data.get("result")
                _on_task_completed(task_name, result)
            return handle

        async def _schedule_tasks():
            ready_tasks_names = []
            with lock:
                for name, deps in remaining_deps.items():
                    if not deps and not futures[name].done() and not futures[name].running():
                        futures[name].set_running_or_notify_cancel()
                        ready_tasks_names.append(name)

            for task_name in ready_tasks_names:
                task = workflow.tasks[task_name]
                dependency_results = {dep: results[dep] for dep in task.dependencies}

                reply_to = f"workflow_result_{uuid.uuid4()}"
                self.message_broker.subscribe(reply_to, _message_handler(task_name))

                message = {
                    "task_name": task.name,
                    "args": dependency_results,
                    "reply_to": reply_to
                }
                self.message_broker.publish(task.name, json.dumps(message))

        await _schedule_tasks()

        await asyncio.gather(*[f for f in futures.values() if not f.done()])

        return results
