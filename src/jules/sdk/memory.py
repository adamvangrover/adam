"""
Jules Memory and Session State Manager.
Manages persistent session history, task logs, and `.jules/` workspace artifacts.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JulesMemoryManager:
    """
    Manages session context, command history, and persistent `.jules/` workspace memories.
    """

    def __init__(self, workspace_root: str | None = None) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.jules_dir = self.workspace_root / ".jules"
        self.jules_dir.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.jules_dir / "memory.json"
        self.session_file = self.jules_dir / "session.json"
        self.history_file = self.jules_dir / "history.jsonl"

    def save_memory(self, key: str, value: Any) -> None:
        """Saves a key-value pair to persistent memory."""
        memories = self.load_memories()
        memories[key] = {
            "value": value,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.memory_file.write_text(json.dumps(memories, indent=2), encoding="utf-8")

    def load_memories(self) -> dict[str, Any]:
        """Loads all persistent memories."""
        if not self.memory_file.exists():
            return {}
        try:
            return json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def log_task_execution(self, prompt: str, plan: list[str], status: str, result: str) -> None:
        """Appends a task execution record to JSONL log."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "prompt": prompt,
            "plan": plan,
            "status": status,
            "result": result,
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Returns recent task execution history records."""
        if not self.history_file.exists():
            return []
        records = []
        with open(self.history_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records[-limit:]
