"""
Jules Tool Registry and Built-In Local Tool System.
Provides safe, auditable execution of file, process, memory, and governance tools.
"""

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JulesToolRegistry:
    """
    Registry for tools that can be invoked by Jules Agent.
    Includes W3C PROV-O lineage tracking for every tool call.
    """

    def __init__(self, workspace_root: str | None = None) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._tools: dict[str, Callable[..., Any]] = {}
        self._descriptions: dict[str, str] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._register_default_tools()

    def register_tool(self, name: str, fn: Callable[..., Any], description: str) -> None:
        """Register a custom tool function with name and description."""
        self._tools[name] = fn
        self._descriptions[name] = description

    def list_tools(self) -> list[dict[str, str]]:
        """Return list of registered tool names and descriptions."""
        return [{"name": name, "description": desc} for name, desc in self._descriptions.items()]

    def execute_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Executes a registered tool by name with arguments and logs W3C PROV-O provenance.
        """
        args = args or {}
        timestamp = datetime.now(UTC).isoformat()

        if tool_name not in self._tools:
            result = {"status": "error", "message": f"Tool '{tool_name}' is not registered."}
        else:
            try:
                fn = self._tools[tool_name]
                out = fn(**args)
                result = {"status": "success", "result": out}
            except Exception as e:
                result = {"status": "error", "message": str(e)}

        provenance_entry = {
            "entity_id": f"urn:jules:tool_exec:{tool_name}:{timestamp}",
            "tool_name": tool_name,
            "args": args,
            "result_status": result["status"],
            "timestamp": timestamp,
            "context_provenance": json.dumps({
                "prov_type": "ToolExecution",
                "tool": tool_name,
                "executed_at": timestamp,
                "agent": "JulesAgent"
            })
        }
        self._audit_log.append(provenance_entry)
        result["provenance"] = provenance_entry
        return result

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return full tool audit history."""
        return self._audit_log

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = (self.workspace_root / p).resolve()
        return p

    def _register_default_tools(self) -> None:
        # 1. list_files
        def list_files(path: str = ".") -> list[str]:
            target = self._resolve_path(path)
            if not target.exists():
                raise FileNotFoundError(f"Directory {path} does not exist.")
            res = []
            for item in sorted(target.iterdir()):
                rel = item.relative_to(self.workspace_root).as_posix()
                res.append(f"{rel}/" if item.is_dir() else rel)
            return res

        self.register_tool("list_files", list_files, "Lists files and directories under path.")

        # 2. read_file
        def read_file(filepath: str) -> str:
            target = self._resolve_path(filepath)
            if not target.exists():
                raise FileNotFoundError(f"File {filepath} not found.")
            return target.read_text(encoding="utf-8")

        self.register_tool("read_file", read_file, "Reads and returns file content.")

        # 3. write_file
        def write_file(filepath: str, content: str) -> str:
            target = self._resolve_path(filepath)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {filepath}."

        self.register_tool("write_file", write_file, "Creates or overwrites file content.")

        # 4. delete_file
        def delete_file(filepath: str) -> str:
            target = self._resolve_path(filepath)
            if target.exists():
                target.unlink()
                return f"Successfully deleted {filepath}."
            return f"File {filepath} did not exist."

        self.register_tool("delete_file", delete_file, "Deletes specified file.")

        # 5. replace_diff
        def replace_diff(filepath: str, search_block: str, replace_block: str) -> str:
            target = self._resolve_path(filepath)
            text = target.read_text(encoding="utf-8")
            if search_block not in text:
                raise ValueError("search_block not found in target file.")
            new_text = text.replace(search_block, replace_block, 1)
            target.write_text(new_text, encoding="utf-8")
            return f"Successfully updated {filepath}."

        self.register_tool("replace_diff", replace_diff, "Replaces search block with replace block in file.")

        # 6. run_bash
        def run_bash(command: str) -> dict[str, Any]:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:2000],
            }

        self.register_tool("run_bash", run_bash, "Runs bash command in workspace directory.")

        # 7. run_pytest
        def run_pytest(test_path: str = "tests/") -> dict[str, Any]:
            cmd = f"uv run pytest {test_path}"
            return run_bash(cmd)

        self.register_tool("run_pytest", run_pytest, "Runs pytest test suite.")

        # 8. git_status
        def git_status() -> dict[str, Any]:
            return run_bash("git status --short")

        self.register_tool("git_status", git_status, "Checks git status of workspace.")

        # 9. read_agents_md
        def read_agents_md() -> str:
            agents_file = self.workspace_root / "AGENTS.md"
            if agents_file.exists():
                return agents_file.read_text(encoding="utf-8")
            return "AGENTS.md file not found."

        self.register_tool("read_agents_md", read_agents_md, "Reads AGENTS.md instructions.")

        # 10. record_memory
        def record_memory(key: str, value: str) -> str:
            mem_dir = self.workspace_root / ".jules"
            mem_dir.mkdir(parents=True, exist_ok=True)
            mem_file = mem_dir / "memory.json"
            data = {}
            if mem_file.exists():
                try:
                    data = json.loads(mem_file.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[key] = {
                "value": value,
                "recorded_at": datetime.now(UTC).isoformat()
            }
            mem_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return f"Recorded memory key '{key}'."

        self.register_tool("record_memory", record_memory, "Saves persistent key-value memory entry.")
