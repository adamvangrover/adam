"""
AFOS v30.1 — Antigravity IDE & MCP Workspace Server
Path A / IDE Ingestion Boundary with AFOS-SNT-01 Path Confinement and W3C PROV-O Auditing.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("adam.antigravity_ide.server")

JSONLOGIC_VERSION: str = "1.4.0"
SENTINEL_AGENT_URN: str = "urn:adam:agent:AFOS-SNT-01"
MAX_FILE_BYTES: int = 5 * 1024 * 1024  # 5 MB deterministic upper bound


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


app = FastAPI(
    title="ADAM Antigravity IDE Workspace Server",
    version="30.1.1",
    docs_url=None,
    redoc_url=None,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "sentinel_gate": "AFOS-SNT-01",
        "jsonLogic_version": JSONLOGIC_VERSION,
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
