"""
Jules Agent & Execution Engine Module.
Coordinates autonomous reasoning loops, step execution, tool calls, and W3C PROV-O audit sealing.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.jules.sdk.config import JulesConfig
from src.jules.sdk.memory import JulesMemoryManager
from src.jules.sdk.providers import BaseLLMProvider, get_llm_provider
from src.jules.sdk.tools import JulesToolRegistry

logger = logging.getLogger("jules.agent")


class JulesAgent:
    """
    Autonomous Jules AI Agent.
    Supports plan creation, tool selection, reasoning, execution, and verification loops.
    """

    def __init__(
        self,
        config: JulesConfig | None = None,
        tool_registry: JulesToolRegistry | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ) -> None:
        self.config = config or JulesConfig()
        self.tool_registry = tool_registry or JulesToolRegistry(self.config.workspace_root)
        self.llm_provider = llm_provider or get_llm_provider(self.config)
        self.memory = JulesMemoryManager(self.config.workspace_root)
        self.current_plan: list[str] = []
        self.conversation_history: list[dict[str, str]] = []

    def set_plan(self, prompt: str) -> list[str]:
        """
        Generates a step-by-step execution plan for the given prompt.
        """
        planning_prompt = (
            f"User Task: {prompt}\n\n"
            "Create a structured numbered plan to accomplish this task safely in the repository. "
            "Format output as bulleted or numbered steps."
        )
        response = self.llm_provider.generate_response(
            messages=[{"role": "user", "content": planning_prompt}],
            system_prompt="You are Jules, an expert software engineering assistant. Formulate clear plans.",
        )
        content = response.get("content", "")
        steps = [line.strip() for line in content.split("\n") if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))]
        if not steps:
            steps = [f"1. Analyze and execute task: {prompt}", "2. Verify changes and run tests."]
        self.current_plan = steps
        return steps

    def execute_step(self, step: str) -> dict[str, Any]:
        """
        Executes a single step, invoking relevant tools if necessary.
        """
        messages = [
            {"role": "user", "content": f"Execute step: {step}\nAvailable tools: {[t['name'] for t in self.tool_registry.list_tools()]}"}
        ]
        response = self.llm_provider.generate_response(
            messages=messages,
            system_prompt="You are Jules. Execute the step and call appropriate tools if required.",
            tools=self.tool_registry.list_tools(),
        )

        tool_results = []
        for tool_call in response.get("tool_calls", []):
            t_name = tool_call.get("name")
            t_args = tool_call.get("args", {})
            out = self.tool_registry.execute_tool(t_name, t_args)
            tool_results.append(out)

        return {
            "step": step,
            "response": response.get("content", ""),
            "tool_results": tool_results,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def run(self, prompt: str) -> dict[str, Any]:
        """
        Autonomous execution loop for a user request.
        Generates plan, iterates through steps, executes actions, and returns task summary.
        """
        logger.info(f"Starting autonomous Jules loop for task: {prompt}")
        plan = self.set_plan(prompt)
        step_outputs = []

        for step in plan:
            out = self.execute_step(step)
            step_outputs.append(out)

        # Verification step
        verify_output = self.tool_registry.execute_tool("git_status", {})

        task_summary = {
            "prompt": prompt,
            "plan": plan,
            "step_outputs": step_outputs,
            "verification": verify_output,
            "status": "completed",
            "audit_log": self.tool_registry.get_audit_log(),
            "context_provenance": json.dumps({
                "agent": "JulesAgent",
                "task": prompt,
                "completed_at": datetime.now(UTC).isoformat(),
                "steps_count": len(plan),
            }),
        }

        self.memory.log_task_execution(prompt, plan, "completed", json.dumps(task_summary.get("verification")))
        return task_summary

    def chat(self, message: str) -> str:
        """
        Interactive chat conversation.
        """
        self.conversation_history.append({"role": "user", "content": message})
        response = self.llm_provider.generate_response(
            messages=self.conversation_history,
            system_prompt="You are Jules, an expert software engineering assistant built into ADAM OS.",
        )
        reply = response.get("content", "")
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply


class JulesEngine:
    """
    High-level factory engine for Jules Agent workflow instantiation.
    """

    @staticmethod
    def create(
        provider: str = "mock",
        model_name: str = "mock-jules-v1",
        workspace_root: str | None = None,
    ) -> JulesAgent:
        cfg = JulesConfig(provider=provider, model_name=model_name, workspace_root=workspace_root or ".")
        return JulesAgent(config=cfg)
