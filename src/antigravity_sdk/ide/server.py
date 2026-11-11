"""
Antigravity IDE Server & Web API.
Provides REST and WebSocket endpoints for file navigation, interactive editing, agent chat, plugin/extension inspection.
"""
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..app import AntigravityApp, AppConfig
from ..plugins import default_plugin_registry
from ..extensions import default_extension_host
from ..runtimes import default_runtime_registry

logger = logging.getLogger("antigravity.ide.server")

app = FastAPI(
    title="Antigravity Web IDE Server",
    description="Interactive Web Workbench and IDE Backend for Antigravity App & Ecosystem",
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
    runtime: str | None = "offline"


class PromptRequest(BaseModel):
    prompt: str
    runtime: str | None = "offline"


class FileSaveRequest(BaseModel):
    filepath: str
    content: str


_app_instance: AntigravityApp | None = None


def get_app(runtime: str = "offline") -> AntigravityApp:
    global _app_instance
    if _app_instance is None or _app_instance.config.runtime_mode != runtime:
        _app_instance = AntigravityApp(AppConfig(runtime_mode=runtime))
    return _app_instance


@app.get("/api/status")
def api_status():
    instance = get_app()
    return instance.get_status()


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    instance = get_app(req.runtime or "offline")
    res = instance.execute_prompt(req.message)
    return {"reply": res["response"], "runtime": instance.config.runtime_mode}


@app.post("/api/run")
def api_run(req: PromptRequest):
    instance = get_app(req.runtime or "offline")
    return instance.execute_prompt(req.prompt)


@app.get("/api/plugins")
def api_list_plugins():
    return default_plugin_registry.list_plugins()


@app.get("/api/extensions")
def api_list_extensions():
    return default_extension_host.list_extensions()


@app.get("/api/runtimes")
def api_list_runtimes():
    return default_runtime_registry.list_runtimes()


@app.get("/api/files")
def api_list_files(path: str = "."):
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    try:
        entries = []
        for item in p.iterdir():
            if item.name.startswith(".venv") or item.name == ".git":
                continue
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "path": str(item.relative_to(Path.cwd())),
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return sorted(entries, key=lambda x: (not x["is_dir"], x["name"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/file")
def api_read_file(filepath: str):
    p = Path(filepath).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        return {"filepath": filepath, "content": p.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/file")
def api_save_file(req: FileSaveRequest):
    p = Path(req.filepath).resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.content, encoding="utf-8")
        return {"status": "saved", "filepath": req.filepath, "bytes": len(req.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
@app.get("/ide", response_class=HTMLResponse)
def get_ide_html():
    ide_path = Path(__file__).parent / "static" / "antigravity_ide.html"
    if ide_path.exists():
        return ide_path.read_text(encoding="utf-8")
    return "<html><body><h1>Antigravity IDE Interface</h1><p>Static bundle loading...</p></body></html>"


def run_server(host: str = "127.0.0.1", port: int = 8888) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")
