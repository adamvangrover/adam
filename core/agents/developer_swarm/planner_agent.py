#planner_agent.py
"""
This module defines the PlannerAgent, a specialized agent responsible for
decomposing high-level tasks into detailed, actionable plans.
"""

from typing import Any, Dict, List, Optional
from core.agents.agent_base import AgentBase

class PlannerAgent(AgentBase):
    """
    The PlannerAgent takes a high-level feature request or bug report
    and breaks it down into a detailed, structured plan with discrete,
    verifiable steps. This plan can then be executed by other agents
    in the developer swarm.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.name = "PlannerAgent"

    async def execute(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Takes a task description and returns a structured plan.

        :param task_description: A string describing the high-level task.
        :return: A list of dictionaries, where each dictionary represents a
                 step in the plan.
        """
        # In a real implementation, this would involve a sophisticated call
        # to an LLM with a carefully crafted prompt to break down the task.
        # For now, this is a placeholder implementation.
        self.context["plan"] = [
            {"step": 1, "task": "Understand the request", "status": "pending"},
            {"step": 2, "task": "Create a plan", "status": "pending"},
            {"step": 3, "task": "Execute the plan", "status": "pending"},
        ]
        return self.context["plan"]

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the skills of the PlannerAgent.
        """
        schema = super().get_skill_schema()
        schema["skills"].append(
            {
                "name": "create_plan",
                "description": "Creates a detailed plan from a high-level task description.",
                "parameters": [
                    {"name": "task_description", "type": "string", "description": "The high-level task description."}
                ]
            }
        )
        return schema
