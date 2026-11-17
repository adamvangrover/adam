"""
AFOS v30.1 — Antigravity IDE & MCP Workspace Server
Path A / IDE Ingestion Boundary with AFOS-SNT-01 Path Confinement and W3C PROV-O Auditing.
AFOS v30.1 — Antigravity IDE, Ecosystem Hub & MCP Workspace Server
Path A / IDE Ingestion Boundary with AFOS-SNT-01 Path Confinement,
Unified Ecosystem Hub endpoints, and W3C PROV-O Auditing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse

    HAS_FASTAPI = True
except ImportError:
    from src.antigravity_ide._web_shim import (
        CORSMiddleware,
        FastAPI,
        HTMLResponse,
        HTTPException,
        status,
    )

    HAS_FASTAPI = False

from src.antigravity_sdk import (
    Agent,
    CapabilitiesConfig,
    JulesAgent,
    LocalAgentConfig,
    ToolRunner,
)
from src.orchestration.ecosystem_hub import EcosystemHub

logger = logging.getLogger("adam.antigravity_ide.server")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
JSONLOGIC_VERSION: str = "1.4.0"
SENTINEL_AGENT_URN: str = "urn:adam:agent:AFOS-SNT-01"
MAX_FILE_BYTES: int = 5 * 1024 * 1024  # 5 MB deterministic upper bound

_hub = EcosystemHub()


def get_allowed_workspace_root() -> Path:
    """
    Returns the canonical allow-listed workspace directory.
    Defaults to ./workspace if ADAM_WORKSPACE_ROOT is not set in the environment.
    """
    raw_root = os.environ.get("ADAM_WORKSPACE_ROOT")
    if raw_root:
        root_path = Path(raw_root)
    else:
        root_path = Path.cwd() / "workspace"
        root_path.mkdir(parents=True, exist_ok=True)
    return root_path.resolve(strict=True)


class SafeWorkspacePathRequest(BaseModel):
    """Strict Pydantic v2 validator blocking traversal primitives at the schema boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Workspace-relative file or directory path.",
    )
    jsonLogic_version: str = Field(default=JSONLOGIC_VERSION)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("AFOS-SNT-01 Block: Null byte injection detected.")

        # Multi-pass URL unquoting to defeat single and double encoding (%2e%2e, %252e%252e)
        unquoted_once = urllib.parse.unquote(value)
        unquoted_twice = urllib.parse.unquote(unquoted_once)

        for candidate in (value, unquoted_once, unquoted_twice):
            if "\x00" in candidate:
                raise ValueError("AFOS-SNT-01 Block: Encoded null byte detected.")
            if "\\" in candidate:
                raise ValueError("AFOS-SNT-01 Block: Backslash path separators are prohibited.")
            if candidate.startswith("/") or (len(candidate) > 1 and candidate[1] == ":"):
                raise ValueError("AFOS-SNT-01 Block: Absolute or drive-qualified paths are prohibited.")
            if ".." in Path(candidate).parts:
                raise ValueError("AFOS-SNT-01 Block: Parent directory traversal ('..') is prohibited.")

        normalized = unquoted_twice.strip()
        if not normalized:
            raise ValueError("AFOS-SNT-01 Block: Empty relative path after normalization.")
        return normalized


class WorkspaceFileReadRequest(SafeWorkspacePathRequest):
    """Request schema for reading a confined workspace file."""


class WorkspaceFileWriteRequest(SafeWorkspacePathRequest):
    """Request schema for writing a confined workspace file."""

    content: str = Field(..., max_length=MAX_FILE_BYTES)
    create_parents: bool = Field(default=False)


class WorkspaceListRequest(SafeWorkspacePathRequest):
    """Request schema for listing a confined workspace directory."""

    relative_path: str = Field(default=".", min_length=1, max_length=512)


class WorkspaceOperationResponse(BaseModel):
    """Canonical AFOS v30.1 response payload with W3C PROV-O provenance."""

    model_config = ConfigDict(extra="forbid")

    status: str
    relative_path: str
    content: Optional[str] = None
    entries: Optional[List[str]] = None
    confidence: float = Field(default=0.94, ge=0.0, le=1.0)
    jsonLogic_version: str = Field(default=JSONLOGIC_VERSION)
    prov_o_audit_trail: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """IDE workbench chat request."""

    message: str = Field(..., min_length=1, max_length=16000)
    model: str = Field(default="gemini-2.5-pro")
    session_id: Optional[str] = Field(default=None)


class TaskRequest(BaseModel):
    """IDE workbench autonomous task / plan request."""

    prompt: str = Field(..., min_length=1, max_length=32000)
    model: str = Field(default="gemini-2.5-pro")


class FileSaveRequest(BaseModel):
    """IDE workbench file save request."""

    filepath: str = Field(..., min_length=1, max_length=1024)
    content: str = Field(..., max_length=MAX_FILE_BYTES)


class OrchestratorRunRequest(BaseModel):
    """Ecosystem orchestrator execution request."""

    orchestrator_id: str = Field(default="gate_clearance")
    params: Dict[str, Any] = Field(default_factory=dict)


class SkillRunRequest(BaseModel):
    """Ecosystem skill execution request."""

    skill_id: str = Field(..., min_length=1)
    payload: Optional[Dict[str, Any]] = Field(default=None)


class UnderwriteRequest(BaseModel):
    """Single-name AI & TMT compute lease underwriting request."""

    model_config = ConfigDict(extra="allow")

    ticker: str = Field(default="NEOC")
    company_name: str = Field(default="Neocloud Compute Infrastructure Corp")
    sector: str = Field(default="TMT / AI Compute Infrastructure")
    revenue_m: float = Field(default=1800.0)
    ebitda_m: float = Field(default=420.0)
    gaap_funded_debt_m: float = Field(default=950.0)
    off_balance_sheet_lease_commitments_m: float = Field(default=3200.0)
    annual_lease_expense_m: float = Field(default=480.0)
    lease_discount_rate_pct: float = Field(default=6.0)
    lease_term_years: float = Field(default=5.0)
    cash_interest_m: float = Field(default=65.0)
    capex_maint_m: float = Field(default=90.0)
    cash_taxes_m: float = Field(default=45.0)
    delta_wc_leases_m: float = Field(default=35.0)
    undrawn_rcf_m: float = Field(default=400.0)
    rcf_commitment_fee_bps: float = Field(default=37.5)
    spot_sofr_pct: float = Field(default=4.35)
    rate_shock_bps: float = Field(default=150.0)
    floating_debt_share: float = Field(default=0.75)
    theta_threshold: float = Field(default=0.03)
    liquidity_runway_days: int = Field(default=240)


def build_prov_o_record(action: str, relative_target: str, confidence: float = 0.94) -> Dict[str, Any]:
    """Generates a deterministic W3C PROV-O activity entry per AFOS v30.1 Invariant B-E."""
    return {
        "@type": "prov:Activity",
        "prov:wasAssociatedWith": SENTINEL_AGENT_URN,
        "prov:used": [f"urn:adam:workspace:{relative_target}"],
        "prov:generated": f"urn:adam:artifact:antigravity_ide:{action}",
        "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
        "jsonLogic_version": JSONLOGIC_VERSION,
        "confidence": confidence,
    }


def _reject_symlinks_in_chain(root: Path, relative_subpath: str) -> None:
    """Walks every segment from root to target and rejects symbolic links prior to I/O."""
    current = root
    for part in Path(relative_subpath).parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="AFOS-SNT-01 Quarantine Block: Symbolic link traversal is prohibited.",
            )


def resolve_confined_workspace_path(
    user_input_path: str,
    allowed_root: Optional[Path] = None,
) -> Path:
    """
    Confines `user_input_path` strictly inside `allowed_root`.
    1. Validates against URL-encoded traversal, null bytes, and absolute paths via Pydantic v2.
    2. Rejects any symbolic links along the path chain.
    3. Sanitizes using `os.path.realpath` + `startswith` + `os.path.commonpath` (CodeQL py/path-injection guard).
    4. Verifies `Path.resolve().is_relative_to()` (AFOS v30.1 Step 5 P0 Security Checklist).
    """
    try:
        validated = SafeWorkspacePathRequest(relative_path=user_input_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    base_root: Path = (allowed_root or get_allowed_workspace_root()).resolve(strict=True)

    # Check intermediate segments for symlink escapes before resolving
    _reject_symlinks_in_chain(base_root, validated.relative_path)

    # CodeQL py/path-injection recognized sanitizer guard
    root_real: str = os.path.realpath(str(base_root))
    joined_candidate: str = os.path.join(root_real, validated.relative_path)
    target_real: str = os.path.realpath(joined_candidate)

    is_exact_root = target_real == root_real
    is_strict_child = target_real.startswith(root_real + os.sep)
    shares_common_root = os.path.commonpath([root_real, target_real]) == root_real

    if not ((is_exact_root or is_strict_child) and shares_common_root):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AFOS-SNT-01 Quarantine Block: Path escapes allow-listed workspace root.",
        )

    # AFOS v30.1 Step 5 pathlib confinement verification
    resolved_target = Path(target_real).resolve(strict=False)
    if not resolved_target.is_relative_to(base_root):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AFOS-SNT-01 Quarantine Block: Resolved path violates workspace boundary.",
        )

    return resolved_target


def _resolve_workbench_path(raw_path: str) -> Path:
    """Confines workbench file access inside Repo Root, Workspace Root, or System Tempdir."""
    if "\x00" in raw_path:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Null byte path blocked.")
    unquoted = urllib.parse.unquote(urllib.parse.unquote(raw_path))
    if ".." in Path(unquoted).parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Parent traversal ('..') blocked.")

    candidate = Path(unquoted)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate)

    if candidate.is_symlink():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Symlink access blocked.")

    resolved = candidate.resolve(strict=False)
    allowed_roots = [
        Path.cwd().resolve(strict=False),
        REPO_ROOT.resolve(strict=False),
        Path(tempfile.gettempdir()).resolve(strict=False),
    ]
    raw_ws = os.environ.get("ADAM_WORKSPACE_ROOT")
    if raw_ws:
        allowed_roots.append(Path(raw_ws).resolve(strict=False))

    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AFOS-SNT-01 Quarantine Block: Workbench path outside allowed roots.",
        )
    return resolved


def _default_ide_tools() -> List[Any]:
    def read_repo_file(filepath: str) -> str:
        """Reads content of a repository file."""
        target = _resolve_workbench_path(filepath)
        if not target.exists() or not target.is_file():
            return f"File not found: {filepath}"
        return target.read_text(encoding="utf-8", errors="replace")

    def list_repo_files(directory: str = ".") -> str:
        """Lists entries in a repository directory."""
        target = _resolve_workbench_path(directory)
        if not target.exists() or not target.is_dir():
            return f"Directory not found: {directory}"
        return "\n".join(sorted(p.name for p in target.iterdir())[:100])

    def run_ecosystem_orchestrator(orchestrator_id: str = "gate_clearance") -> str:
        """Executes a registered AFOS v30.1-R orchestrator pipeline."""
        return json.dumps(_hub.run_orchestrator(orchestrator_id), default=str)

    return [read_repo_file, list_repo_files, run_ecosystem_orchestrator]


app = FastAPI(
    title="ADAM Antigravity IDE Workspace Server",
    title="ADAM Antigravity IDE & Ecosystem Server",
    version="30.1.1",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ---------------------------------------------------------------------------
# Confined Workspace IDE v1 Security Endpoints (AFOS-SNT-01)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "sentinel_gate": "AFOS-SNT-01",
        "jsonLogic_version": JSONLOGIC_VERSION,
        "ecosystem_hub": "online",
    }


@app.post("/api/v1/ide/read", response_model=WorkspaceOperationResponse)
async def read_workspace_file(payload: WorkspaceFileReadRequest) -> WorkspaceOperationResponse:
    root = get_allowed_workspace_root()
    safe_target = resolve_confined_workspace_path(payload.relative_path, allowed_root=root)

    if not safe_target.exists() or not safe_target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested workspace file does not exist.",
        )

    if safe_target.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Workspace file exceeds maximum allowed size bound.",
        )

    file_text = safe_target.read_text(encoding="utf-8")
    rel_str = str(safe_target.relative_to(root))

    return WorkspaceOperationResponse(
        status="ok",
        relative_path=rel_str,
        content=file_text,
        prov_o_audit_trail=[build_prov_o_record("read_file", rel_str)],
    )


@app.post("/api/v1/ide/write", response_model=WorkspaceOperationResponse)
async def write_workspace_file(payload: WorkspaceFileWriteRequest) -> WorkspaceOperationResponse:
    root = get_allowed_workspace_root()
    safe_target = resolve_confined_workspace_path(payload.relative_path, allowed_root=root)

    if safe_target == root or safe_target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot overwrite workspace directory root.",
        )

    parent_dir = safe_target.parent
    if not parent_dir.is_relative_to(root):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AFOS-SNT-01 Quarantine Block: Parent directory outside workspace root.",
        )

    if payload.create_parents:
        parent_dir.mkdir(parents=True, exist_ok=True)
    elif not parent_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target parent directory does not exist.",
        )

    safe_target.write_text(payload.content, encoding="utf-8")
    rel_str = str(safe_target.relative_to(root))

    return WorkspaceOperationResponse(
        status="written",
        relative_path=rel_str,
        prov_o_audit_trail=[build_prov_o_record("write_file", rel_str)],
    )


@app.post("/api/v1/ide/list", response_model=WorkspaceOperationResponse)
async def list_workspace_directory(payload: WorkspaceListRequest) -> WorkspaceOperationResponse:
    root = get_allowed_workspace_root()
    safe_target = resolve_confined_workspace_path(payload.relative_path, allowed_root=root)

    if not safe_target.exists() or not safe_target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested workspace directory does not exist.",
        )

    entries = sorted(
        entry.name + ("/" if entry.is_dir() else "")
        for entry in safe_target.iterdir()
        if not entry.is_symlink()
    )
    rel_str = "." if safe_target == root else str(safe_target.relative_to(root))

    return WorkspaceOperationResponse(
        status="ok",
        relative_path=rel_str,
        entries=entries,
        prov_o_audit_trail=[build_prov_o_record("list_dir", rel_str)],
    )


# ---------------------------------------------------------------------------
# Antigravity Workbench & Ecosystem Hub API Handlers
# ---------------------------------------------------------------------------


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    return {
        "status": "online",
        "version": "30.1.1",
        "sentinel_gate": "AFOS-SNT-01",
        "workspace_root": str(Path.cwd()),
        "capabilities": {
            "file_system_write": True,
            "terminal": True,
            "code_execution": True,
            "ecosystem_orchestrators": True,
            "credit_underwriting": True,
        },
        "orchestrator_count": len(_hub.list_orchestrators()),
        "skill_count": len(_hub.list_skills()),
    }


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> Dict[str, Any]:
    session_id = req.session_id or f"agy-sess-{uuid.uuid4().hex[:8]}"
    config = LocalAgentConfig(
        model_name=req.model,
        capabilities=CapabilitiesConfig(allow_terminal=True, allow_code_execution=True),
        tools=_default_ide_tools(),
    )
    async with Agent(config) as agent:
        resp = await agent.chat(req.message)
        reply_text = await resp.text()
    return {
        "reply": reply_text,
        "session_id": session_id,
        "model": req.model,
        "prov_o_audit_trail": [build_prov_o_record("ide_chat", session_id)],
    }


@app.post("/api/plan")
async def api_plan(req: TaskRequest) -> Dict[str, Any]:
    plan = [
        f"1. [SYSTEM 1 SWARM] Ingest prompt & scan bitemporal state for: {req.prompt[:80]}",
        "2. [AFOS-SNT-01] Enforce Lexicographic Layer 0 x Layer 1 security & path gates.",
        "3. [SYSTEM 2 DAG] Route calculations through deterministic EcosystemHub kernels.",
        "4. [AFOS-CMP-01] Arbitrate dual-model divergence & verify 75/25 Fortress allocation.",
        "5. [W3C PROV-O] Seal reasoning lineage and emit SHA-256 verification manifest.",
    ]
    return {
        "prompt": req.prompt,
        "model": req.model,
        "routing_path": "SYSTEM_2_DAG",
        "plan": plan,
    }


@app.post("/api/run")
async def api_run(req: TaskRequest) -> Dict[str, Any]:
    jules = JulesAgent()
    return await jules.execute_task(req.prompt)


@app.get("/api/files")
def api_list_files(path: str = ".") -> Dict[str, Any]:
    target = _resolve_workbench_path(path)
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    entries: List[Dict[str, Any]] = []
    cwd_resolved = Path.cwd().resolve()
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if item.name.startswith(".") or item.name in ("__pycache__", "node_modules") or item.is_symlink():
            continue
        try:
            rel = str(item.resolve().relative_to(cwd_resolved))
        except ValueError:
            rel = str(item)
        entries.append(
            {
                "name": item.name,
                "path": rel.replace("\\", "/"),
                "is_dir": item.is_dir(),
            }
        )
        if len(entries) >= 200:
            break

    return {"status": "ok", "path": path, "entries": entries}


@app.get("/api/file")
def api_read_file(filepath: str) -> Dict[str, Any]:
    target = _resolve_workbench_path(filepath)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds MAX_FILE_BYTES limit.")
    return {
        "status": "ok",
        "filepath": filepath,
        "content": target.read_text(encoding="utf-8", errors="replace"),
    }


@app.post("/api/file")
def api_save_file(req: FileSaveRequest) -> Dict[str, Any]:
    target = _resolve_workbench_path(req.filepath)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is a directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    return {
        "status": "saved",
        "filepath": req.filepath,
        "bytes_written": len(req.content.encode("utf-8")),
    }


@app.get("/api/tools")
def api_list_tools() -> Dict[str, Any]:
    runner = ToolRunner(tools=_default_ide_tools())
    return {
        "tools": runner.get_tool_schemas(),
        "skills": _hub.list_skills(),
    }


@app.get("/api/triggers")
def api_list_triggers() -> Dict[str, Any]:
    return {
        "active_triggers": 1,
        "default_interval_seconds": 30.0,
        "events": [
            {
                "trigger_id": "prov_o_bitemporal_watcher",
                "type": "periodic",
                "interval_seconds": 30.0,
                "status": "armed",
            }
        ],
    }


@app.get("/api/extensions")
def api_list_extensions() -> Dict[str, Any]:
    extensions: List[Dict[str, Any]] = []
    ext_root = REPO_ROOT / "extensions"
    if ext_root.exists():
        for pkg in sorted(ext_root.glob("**/package.json")):
            try:
                extensions.append(json.loads(pkg.read_text(encoding="utf-8")))
            except Exception:
                continue

    plugins: List[Dict[str, Any]] = []
    plugins_root = REPO_ROOT / "plugins"
    if plugins_root.exists():
        for pf in sorted(plugins_root.glob("*.json")):
            try:
                plugins.append(json.loads(pf.read_text(encoding="utf-8")))
            except Exception:
                continue

    return {"extensions": extensions, "plugins": plugins}


@app.get("/api/ecosystem")
def api_get_ecosystem() -> Dict[str, Any]:
    return _hub.get_ecosystem_manifest()


@app.post("/api/orchestrators/run")
def api_run_orchestrator(req: OrchestratorRunRequest) -> Dict[str, Any]:
    try:
        return _hub.run_orchestrator(req.orchestrator_id, req.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/skills/run")
def api_run_skill(req: SkillRunRequest) -> Dict[str, Any]:
    try:
        return _hub.run_skill(req.skill_id, req.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/underwrite")
def api_underwrite(req: UnderwriteRequest) -> Dict[str, Any]:
    return _hub.underwrite_obligor(req.model_dump())


@app.get("/api/security/audit")
def api_security_audit() -> Dict[str, Any]:
    return _hub.run_security_ci_gate()


def _serve_repo_html(relative_path: str, fallback_title: str) -> HTMLResponse:
    target = (REPO_ROOT / relative_path).resolve(strict=False)
    if target.is_relative_to(REPO_ROOT.resolve()) and target.exists() and target.is_file():
        return HTMLResponse(content=target.read_text(encoding="utf-8", errors="replace"))
    return HTMLResponse(content=f"<html><body><h1>{fallback_title}</h1><p>File not found: {relative_path}</p></body></html>", status_code=404)


@app.get("/", response_class=HTMLResponse)
@app.get("/ide", response_class=HTMLResponse)
def get_ide_html() -> HTMLResponse:
    return _serve_repo_html("src/antigravity_ide/static/antigravity_ide.html", "Antigravity Agent IDE")


@app.get("/command-center", response_class=HTMLResponse)
@app.get("/showcase/command_center_v30_1.html", response_class=HTMLResponse)
def get_command_center_html() -> HTMLResponse:
    return _serve_repo_html("showcase/command_center_v30_1.html", "AFOS v30.1 Ecosystem Command Center")


@app.get("/index.html", response_class=HTMLResponse)
def get_root_index_html() -> HTMLResponse:
    return _serve_repo_html("index.html", "ADAM v30.1 OS Root")


@app.get("/showcase/{filename}", response_class=HTMLResponse)
def get_showcase_page(filename: str) -> HTMLResponse:
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".html"):
        raise HTTPException(status_code=403, detail="Invalid showcase filename.")
    return _serve_repo_html(f"showcase/{filename}", f"ADAM Showcase: {filename}")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    if HAS_FASTAPI:
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        logger.info("FastAPI not installed; starting AFOS zero-dependency stdlib server on %s:%s", host, port)
        app.serve_stdlib(host=host, port=port)
