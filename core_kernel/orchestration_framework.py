"""
Core Orchestration Framework.
Provides the event bus and task execution graph for cognitive domains.
"""

import logging
import uuid
from typing import Dict, Any, Callable, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrchestrationEngine")

class OrchestratorEngine:
    def __init__(self):
        self.task_registry: Dict[str, Callable] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def register_skill(self, skill_name: str, executable: Callable):
        """Register a domain skill with the central orchestrator."""
        self.task_registry[skill_name] = executable
        logger.info(f"Registered skill: {skill_name}")

    def execute_skill(self, skill_name: str, payload: Any, context: dict = None) -> Dict[str, Any]:
        """Execute a registered skill and record provenance."""
        if skill_name not in self.task_registry:
            raise ValueError(f"Skill '{skill_name}' is not registered.")

        execution_id = str(uuid.uuid4())
        logger.info(f"Executing '{skill_name}' [Execution ID: {execution_id}]")

        try:
            result = self.task_registry[skill_name](payload)
            record = {
                "execution_id": execution_id,
                "skill": skill_name,
                "status": "SUCCESS",
                "context": context or {},
                # In a real system, we'd hash the inputs/outputs here
            }
            self.execution_history.append(record)
            return {"status": "success", "data": result, "provenance": record}
        except Exception as e:
            record = {
                "execution_id": execution_id,
                "skill": skill_name,
                "status": "FAILED",
                "error": str(e)
            }
            self.execution_history.append(record)
            logger.error(f"Execution failed: {str(e)}")
            return {"status": "error", "message": str(e), "provenance": record}
