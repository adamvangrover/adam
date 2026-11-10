"""
Antigravity IDE & Workbench FastAPI Server.
Provides backend REST & WebSocket endpoints for agent chat, task running,
planning, file browser & editor, tool execution, background triggers, and extensions.
"""

import json
import logging
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.antigravity_sdk import Agent, JulesAgent, LocalAgentConfig
from src.antigravity_sdk.tools import ToolRunner
from src.antigravity_sdk.triggers import TriggerRunner, every

logger = logging.getLogger("antigravity.ide.server")

app = FastAPI(
    title="Antigravity Agent IDE & Workbench Server",
    description="Interactive Agent Workbench, Runtime Backend, and Code Editor for Antigravity",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-2.5-pro"


class TaskRequest(BaseModel):
    prompt: str
    model: str = "gemini-2.5-pro"


class FileSaveRequest(BaseModel):
    filepath: str
    content: str


# In-memory trigger simulation log
_trigger_events: list[dict[str, Any]] = []


def _record_trigger_event(ctx: Any) -> None:
    _trigger_events.append({"status": "fired", "type": "heartbeat_trigger"})


_trigger_runner = TriggerRunner(triggers=[every(30.0, _record_trigger_event)])


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Returns Antigravity runtime, agent capabilities, and system health status."""
    return {
        "status": "online",
        "system": "Antigravity Runtime Environment (AFOS v30.1)",
        "version": "1.0.0",
        "capabilities": {
            "file_system_write": True,
            "code_execution": True,
            "terminal": True,
            "mcp_stdio": True,
        },
        "safety_mode": "Lexicographic L0 x L1 Gating",
        "triggers_active": len(_trigger_runner.triggers),
        "trigger_events_count": len(_trigger_events),
    }


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> dict[str, Any]:
    """Processes interactive chat prompt with the Antigravity agent."""
    config = LocalAgentConfig(model_name=req.model)
    async with Agent(config) as agent:
        response = await agent.chat(req.message)
        text = await response.text()
        return {
            "reply": text,
            "model": req.model,
            "session_id": agent.conversation.session_id,
        }


@app.post("/api/run")
async def api_run(req: TaskRequest) -> dict[str, Any]:
    """Executes an autonomous task prompt using the Jules / Antigravity agent."""
    jules = JulesAgent()
    result = await jules.execute_task(req.prompt)
    return result


@app.post("/api/plan")
async def api_plan(req: TaskRequest) -> dict[str, Any]:
    """Generates a structured execution plan for a prompt."""
    plan = [
        f"1. Analyze objective: '{req.prompt}' and identify repository dependencies.",
        "2. Inspect existing modules, verify interfaces, and check unit tests.",
        "3. Implement additive modifications enforcing PROV-O compliance.",
        "4. Validate results against test suite and record audit provenance.",
    ]
    return {
        "prompt": req.prompt,
        "plan": plan,
        "model": req.model,
    }


@app.get("/api/files")
def api_list_files(path: str = ".") -> dict[str, Any]:
    """Lists files and directories under the given workspace path."""
    target = Path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    entries = []
    try:
        for item in sorted(target.iterdir()):
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"path": path, "entries": entries}


@app.get("/api/file")
def api_read_file(filepath: str) -> dict[str, Any]:
    """Reads file content from the workspace."""
    p = Path(filepath)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    try:
        content = p.read_text(encoding="utf-8")
        return {"filepath": filepath, "content": content, "size": len(content)}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Binary or non-UTF-8 file cannot be displayed")


@app.post("/api/file")
def api_save_file(req: FileSaveRequest) -> dict[str, Any]:
    """Saves file content to the workspace."""
    p = Path(req.filepath)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.content, encoding="utf-8")
        return {"status": "saved", "filepath": req.filepath, "bytes_written": len(req.content)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tools")
def api_list_tools() -> dict[str, Any]:
    """Lists registered tools in the Antigravity tool registry."""
    def dummy_bash_exec(command: str) -> str:
        """Executes a bash command in sandbox."""
        return f"Executed: {command}"

    def dummy_read_file(filepath: str) -> str:
        """Reads file content."""
        return f"Content of {filepath}"

    runner = ToolRunner(tools=[dummy_bash_exec, dummy_read_file])
    return {"tools": runner.get_tool_schemas()}


@app.get("/api/triggers")
def api_list_triggers() -> dict[str, Any]:
    """Returns active background triggers and telemetry."""
    return {
        "active_triggers": len(_trigger_runner.triggers),
        "events": _trigger_events,
        "default_interval_seconds": 30.0,
    }


@app.get("/api/extensions")
def api_list_extensions() -> dict[str, Any]:
    """Lists installed IDE extensions and plugins."""
    extensions = []
    ext_dir = Path("extensions")
    if ext_dir.exists():
        for pkg in ext_dir.glob("**/package.json"):
            try:
                data = json.loads(pkg.read_text(encoding="utf-8"))
                extensions.append(data)
            except Exception:
                pass

    plugins = []
    plugins_dir = Path("plugins")
    if plugins_dir.exists():
        for plg in plugins_dir.glob("*.json"):
            try:
                pdata = json.loads(plg.read_text(encoding="utf-8"))
                plugins.append(pdata)
            except Exception:
                pass

    return {"extensions": extensions, "plugins": plugins}


@app.get("/", response_class=HTMLResponse)
@app.get("/ide", response_class=HTMLResponse)
def get_ide_html() -> str:
    """Renders the Antigravity IDE Workbench Web interface."""
    ide_path = Path(__file__).parent / "static" / "antigravity_ide.html"
    if ide_path.exists():
        return ide_path.read_text(encoding="utf-8")
    return "<html><body><h1>Antigravity IDE Workbench</h1><p>UI loading...</p></body></html>"


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Runs the FastAPI server with uvicorn."""
    uvicorn.run(app, host=host, port=port, log_level="info")
