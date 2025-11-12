# core/agents/orchestrators/meta_orchestrator.py
import asyncio
from typing import Any, Dict, Union

# v21 Synchronous imports
from .workflow import Workflow
from .workflow_manager import WorkflowManager

# v22 Asynchronous imports
from core.system.v22_async.workflow import AsyncWorkflow
from core.system.v22_async.async_workflow_manager import AsyncWorkflowManager

# v23 Graph-based imports
from langgraph.graph import StateGraph

class MetaOrchestrator:
    def __init__(self):
        self.sync_manager = WorkflowManager()
        self.async_manager = AsyncWorkflowManager.get_instance()

    def execute_workflow(
        self,
        workflow: Union[Workflow, AsyncWorkflow, Any],
        initial_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Executes a workflow using the appropriate manager based on its type.
        """
        if isinstance(workflow, AsyncWorkflow):
            # v22 Asynchronous Execution
            return asyncio.run(self.async_manager.execute_workflow(workflow))
        elif isinstance(workflow, Workflow):
            # v21 Synchronous Execution
            return self.sync_manager.execute_workflow(workflow)
        elif hasattr(workflow, 'invoke') and hasattr(workflow, 'graph') and isinstance(workflow.graph, StateGraph):
            # v23 Graph-based Execution
            if initial_state is None:
                initial_state = {}
            return workflow.invoke(initial_state)
        else:
            raise TypeError(f"Unsupported workflow type: {type(workflow)}")
