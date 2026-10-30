"""
Jules Replacement Agent built on top of Antigravity SDK for autonomous software engineering and repository refactoring.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import subprocess

from .agent import Agent
from .config import LocalAgentConfig, CapabilitiesConfig
from .hooks import allow, deny


def list_directory(path: str = ".") -> List[str]:
    """Lists files in the specified repository directory."""
    p = Path(path)
    if not p.exists():
        return [f"Directory {path} does not exist"]
    return [str(item) for item in p.iterdir()]


def read_file_content(filepath: str) -> str:
    """Reads and returns file content."""
    p = Path(filepath)
    if not p.exists():
        return f"File {filepath} not found."
    return p.read_text(encoding="utf-8", errors="replace")


def write_file_content(filepath: str, content: str) -> str:
    """Overwrites or creates a file with the provided content."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Successfully written {len(content)} characters to {filepath}"


def run_unit_tests(test_path: str = "tests/unit/") -> str:
    """Executes pytest unit test suite and returns output summary."""
    try:
        res = subprocess.run(["uv", "run", "pytest", test_path], capture_output=True, text=True, timeout=60)
        return f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except Exception as e:
        return f"Failed to execute tests: {str(e)}"


class JulesAgent:
    """
    Autonomous Jules replacement agent for codebase maintenance, refactoring, and W3C PROV-O audit trailing.
    """

    def __init__(self, system_instructions: Optional[str] = None) -> None:
        instructions = system_instructions or (
            "You are Jules, an autonomous software engineering agent. "
            "Your objective is to inspect, refactor, and maintain code bases while preserving W3C PROV-O "
            "compliance, type safety, and test-driven verification."
        )

        config = LocalAgentConfig(
            system_instructions=instructions,
            capabilities=CapabilitiesConfig(
                read_only=False,
                allow_code_execution=True,
                allow_terminal=True,
                allow_file_system_write=True,
            ),
            tools=[
                list_directory,
                read_file_content,
                write_file_content,
                run_unit_tests,
            ],
            policies=[
                deny("rm -rf *"),
                allow("*"),
            ],
        )

        self.agent = Agent(config=config)
        self.prov_log: List[Dict[str, Any]] = []

    def _record_telemetry(self, activity: str, details: Dict[str, Any]) -> None:
        """Records a W3C PROV-O compliant telemetry log."""
        self.prov_log.append({
            "prov:Activity": activity,
            "prov:Agent": "JulesAgentReplacement",
            "prov:generatedAtTime": datetime.now(timezone.utc).isoformat(),
            "details": details,
        })

    async def execute_task(self, prompt: str) -> Dict[str, Any]:
        """
        Executes a software engineering task autonomously.
        """
        self._record_telemetry("TaskStart", {"prompt": prompt})

        async with self.agent as active_agent:
            chat_response = await active_agent.chat(prompt)
            output_text = await chat_response.text()

        self._record_telemetry("TaskComplete", {"output_length": len(output_text)})

        return {
            "status": "SUCCESS",
            "response": output_text,
            "prov_o_audit_trail": self.prov_log,
        }
