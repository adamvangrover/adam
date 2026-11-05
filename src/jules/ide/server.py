"""
Jules Web IDE REST API & Web Server.
Provides backend endpoints for interactive chat, file browser/editor, agent task execution, and audit trail.
"""

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.jules.sdk.agent import JulesAgent
from src.jules.sdk.config import JulesConfig

logger = logging.getLogger("jules.ide.server")

app = FastAPI(
    title="Jules Agent IDE Server",
    description="Interactive Web Workbench and IDE Backend for Jules Agent",
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
    provider: str | None = "mock"


class TaskRequest(BaseModel):
    prompt: str
    provider: str | None = "mock"


class FileSaveRequest(BaseModel):
    filepath: str
    content: str


# Global agent instance
_agent_instance: JulesAgent | None = None


def get_agent(provider: str = "mock") -> JulesAgent:
    global _agent_instance
    if _agent_instance is None or _agent_instance.config.provider != provider:
        cfg = JulesConfig(provider=provider)
        _agent_instance = JulesAgent(config=cfg)
    return _agent_instance


@app.get("/api/status")
def get_status():
    agent = get_agent()
    cfg = agent.config
    tools = agent.tool_registry.list_tools()
    return {
        "status": "online",
        "version": "1.0.0",
        "provider": cfg.provider,
        "model": cfg.model_name,
        "workspace_root": cfg.workspace_root,
        "tools": [t["name"] for t in tools],
        "audit_logs_count": len(agent.tool_registry.get_audit_log()),
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    agent = get_agent(req.provider or "mock")
    reply = agent.chat(req.message)
    return {"reply": reply, "history": agent.conversation_history}


@app.post("/api/plan")
def api_plan(req: TaskRequest):
    agent = get_agent(req.provider or "mock")
    plan = agent.set_plan(req.prompt)
    return {"prompt": req.prompt, "plan": plan}


@app.post("/api/run")
def api_run(req: TaskRequest):
    agent = get_agent(req.provider or "mock")
    res = agent.run(req.prompt)
    return res


@app.get("/api/files")
def api_list_files(path: str = "."):
    agent = get_agent()
    try:
        files = agent.tool_registry.execute_tool("list_files", {"path": path})
        return files
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/file")
def api_read_file(filepath: str):
    agent = get_agent()
    try:
        content = agent.tool_registry.execute_tool("read_file", {"filepath": filepath})
        return content
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/file")
def api_save_file(req: FileSaveRequest):
    agent = get_agent()
    try:
        res = agent.tool_registry.execute_tool("write_file", {"filepath": req.filepath, "content": req.content})
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
@app.get("/ide", response_class=HTMLResponse)
def get_ide_html():
    ide_path = Path(__file__).parent / "static" / "jules_ide.html"
    if ide_path.exists():
        return ide_path.read_text(encoding="utf-8")
    root_ide = Path.cwd() / "jules_ide.html"
    if root_ide.exists():
        return root_ide.read_text(encoding="utf-8")
    return "<html><body><h1>Jules IDE Interface</h1><p>HTML template loading...</p></body></html>"


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")
