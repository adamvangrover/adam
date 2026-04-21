import asyncio
import logging
from typing import Dict, Any, Callable, Optional
from enum import Enum

logger = logging.getLogger("SovereignWorkflow")

class RunMode(Enum):
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
    MANUAL = "manual"

class SovereignWorkflowConfig:
    def __init__(self, mode: RunMode = RunMode.AUTONOMOUS,
                 whitelist: list = None,
                 parameters: dict = None,
                 human_approval_callback: Optional[Callable] = None):
        self.mode = mode
        self.whitelist = whitelist or []
        self.parameters = parameters or {}
        self.human_approval_callback = human_approval_callback

class SovereignWorkflowEngine:
    """
    Executes a Swarm workflow based on defined execution modes.
    Gracefully degrades to mock outputs if the primary execution fails or is mocked.
    """
    def __init__(self, orchestrator, config: SovereignWorkflowConfig):
        self.orchestrator = orchestrator
        self.config = config

    async def execute_task(self, task: str) -> Dict[str, Any]:
        logger.info(f"Executing task: {task} in mode: {self.config.mode.value}")

        if self.config.mode == RunMode.MANUAL:
            if not self.config.human_approval_callback:
                raise ValueError("Manual mode requires a human_approval_callback.")
            approved = await self.config.human_approval_callback(task)
            if not approved:
                return {"status": "rejected", "reason": "Human denied execution."}

        elif self.config.mode == RunMode.SUPERVISED:
            # Check whitelist and parameters
            if task not in self.config.whitelist:
                if self.config.human_approval_callback:
                    approved = await self.config.human_approval_callback(task)
                    if not approved:
                        return {"status": "rejected", "reason": "Task not in whitelist and human denied."}
                else:
                    return {"status": "rejected", "reason": "Task not in whitelist and no callback provided."}

        # Autonomous or approved execution
        try:
            # Delegate to orchestrator
            result = await self.orchestrator.delegate_task(task, self.config.parameters)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Task execution failed, degrading to mock. Error: {e}")
            return self._graceful_fallback(task)

    def _graceful_fallback(self, task: str) -> Dict[str, Any]:
        """Fallback to mock execution if live logic fails."""
        logger.warning("Applying graceful mock degradation.")
        from core.llm_plugin import MockLLM
        mock_engine = MockLLM()
        mock_response = mock_engine.generate_text(f"Simulate workflow execution for task: {task}")
        return {
            "status": "success_mocked",
            "result": {
                "generated_output": mock_response,
                "note": "This is a vaporware fallback output."
            }
        }
