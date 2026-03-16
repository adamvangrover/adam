# core/agents/orchestrators/hybrid_orchestrator.py
import asyncio
from typing import Any, Dict, Union

from core.system.v22_async.async_workflow_manager import AsyncWorkflowManager
from core.system.v22_async.workflow import AsyncWorkflow

from .workflow import Workflow
from .workflow_manager import WorkflowManager


class HybridOrchestrator:
    def __init__(self):
        self.sync_manager = WorkflowManager()
        self.async_manager = AsyncWorkflowManager.get_instance()

    def execute_workflow(self, workflow: Union[Workflow, AsyncWorkflow]) -> Dict[str, Any]:
        """
        Executes a workflow using the appropriate manager.
        """
        if isinstance(workflow, AsyncWorkflow):
            return asyncio.run(self.async_manager.execute_workflow(workflow))
        elif isinstance(workflow, Workflow):
            return self.sync_manager.execute_workflow(workflow)
        else:
            raise TypeError("Unsupported workflow type")
