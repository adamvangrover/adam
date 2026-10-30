"""
AFOS v30.1 Architect Agent (Jules) — `architect_jules`
======================================================
Implements the Repository Architect agent defined in `AGENTS.md` and `AGENT_CATALOG.md`,
and provides a deterministic command builder and workflow bridge for the `@google/jules`
CLI (`jules`).

Contractual Constraints:
1. Maintain strict isolation between UI presentation layers (`frontend/`, `services/webapp/`,
   `showcase/`) and backend async execution loops (`core/`, `adam_*/`).
2. Enforce Path A vs. Path B bifurcation (`experimental/` never imported into `core/`).
3. Never perform domain arithmetic internally—defer math to deterministic execution kernels.
4. Emit W3C PROV-O compliant `ProvenanceHeader` telemetry on all operations.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

from src.pdil.models import ProvenanceHeader


ALLOWED_THEMES = ("dark", "light")
ALLOWED_SHELLS = ("bash", "zsh", "fish", "powershell")
SAFE_REPO_PATTERN = re.compile(r"^(\.|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)$")
SAFE_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ArchitectJulesState(BaseModel):
    """
    Canonical state schema for `architect_jules` as defined in `AGENTS.md` Section 4.A
    and `AGENT_CATALOG.md`.
    """

    agent_role: Literal["architect_jules"] = Field(
        default="architect_jules",
        description="Canonical role identifier for the Architect Agent (Jules).",
    )
    active_bounded_context: str = Field(
        default="v30.1_async_migration",
        description="Active architectural bounded context.",
    )
    frontend_sync_status: Literal["decoupled", "violation_detected"] = Field(
        default="decoupled",
        description="Isolation status between UI presentation layers and core async execution.",
    )
    last_checkpoint_hash: str = Field(
        default="v30.1-HEAD",
        description="Git commit or state checkpoint hash.",
    )
    prov_o_audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="W3C PROV-O audit trail entries.",
    )


class JulesSessionRequest(BaseModel):
    """
    Validated configuration for creating or managing a `@google/jules` CLI session.
    """

    subcommand: Literal[
        "version",
        "help",
        "login",
        "logout",
        "tui",
        "remote_list_repo",
        "remote_list_session",
        "remote_new",
        "remote_pull",
        "completion",
    ] = Field(default="remote_new", description="Jules CLI operation to execute.")
    repo: Optional[str] = Field(
        default="adamvangrover/adam",
        description="Target repository (e.g., 'adamvangrover/adam', 'torvalds/linux', or '.').",
    )
    session_prompt: Optional[str] = Field(
        default=None,
        description="Task prompt passed to `--session` for `jules remote new`.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier passed to `--session` for `jules remote pull`.",
    )
    parallel: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Number of parallel sessions to launch (`--parallel <number>`).",
    )
    theme: Optional[Literal["dark", "light"]] = Field(
        default=None,
        description="Optional global TUI theme flag (`--theme dark|light`).",
    )
    shell: Optional[Literal["bash", "zsh", "fish", "powershell"]] = Field(
        default="zsh",
        description="Target shell for `jules completion <shell>`.",
    )

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not SAFE_REPO_PATTERN.match(cleaned):
            raise ValueError(
                f"Invalid repository identifier '{value}'. Expected '<owner>/<repo>' or '.'."
            )
        return cleaned

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not SAFE_SESSION_ID_PATTERN.match(cleaned):
            raise ValueError(
                f"Invalid session_id '{value}'. Only alphanumeric characters, '-', and '_' are allowed."
            )
        return cleaned

    @field_validator("session_prompt")
    @classmethod
    def validate_session_prompt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("session_prompt cannot be empty or whitespace.")
        return cleaned


class JulesCommandBuilder:
    """
    Deterministic builder for `@google/jules` CLI invocations.
    Constructs argument arrays safely without shell interpolation.
    """

    BINARY = "jules"

    @classmethod
    def build_args(cls, request: JulesSessionRequest) -> List[str]:
        args: List[str] = [cls.BINARY]

        if request.theme is not None:
            if request.theme not in ALLOWED_THEMES:
                raise ValueError(f"Unsupported theme: {request.theme}")
            args.extend(["--theme", request.theme])

        sub = request.subcommand
        if sub == "tui":
            return args
        if sub == "version":
            args.append("version")
            return args
        if sub == "help":
            args.append("help")
            return args
        if sub == "login":
            args.append("login")
            return args
        if sub == "logout":
            args.append("logout")
            return args
        if sub == "remote_list_repo":
            args.extend(["remote", "list", "--repo"])
            return args
        if sub == "remote_list_session":
            args.extend(["remote", "list", "--session"])
            return args
        if sub == "remote_new":
            if not request.session_prompt:
                raise ValueError("`session_prompt` is required for `remote_new`.")
            args.extend(["remote", "new"])
            if request.repo:
                args.extend(["--repo", request.repo])
            args.extend(["--session", request.session_prompt])
            if request.parallel is not None:
                args.extend(["--parallel", str(request.parallel)])
            return args
        if sub == "remote_pull":
            if not request.session_id:
                raise ValueError("`session_id` is required for `remote_pull`.")
            args.extend(["remote", "pull", "--session", request.session_id])
            return args
        if sub == "completion":
            shell_name = request.shell or "bash"
            if shell_name not in ALLOWED_SHELLS:
                raise ValueError(f"Unsupported shell for completion: {shell_name}")
            args.extend(["completion", shell_name])
            return args

        raise ValueError(f"Unsupported subcommand: {sub}")

    @classmethod
    def format_cli_string(cls, request: JulesSessionRequest) -> str:
        """
        Formats the validated command list as a human-readable, shell-escaped command line.
        """
        args = cls.build_args(request)
        return " ".join(shlex.quote(part) for part in args)


class ArchitectAgent:
    """
    AFOS v30.1 Architect Agent (`architect_jules`).
    Enforces repository architectural invariants and orchestrates `@google/jules` coding sessions.
    """

    ALLOWED_TOOLS = ("git_tools", "static_analyzer", "repo_compiler", "jules_remote_session")
    FORBIDDEN_CORE_UI_IMPORTS = ("streamlit", "react", "dash", "gradio")
    FORBIDDEN_LAB_PREFIXES = ("experimental.", "research.", "tinker_lab.")
    FORBIDDEN_P0_CALLS = ("pickle.load(", "eval(", "exec(")

    def __init__(
        self,
        active_bounded_context: str = "v30.1_async_migration",
        last_checkpoint_hash: str = "v30.1-HEAD",
    ) -> None:
        self.state = ArchitectJulesState(
            active_bounded_context=active_bounded_context,
            last_checkpoint_hash=last_checkpoint_hash,
        )

    def audit_code_snippet(self, code: str, module_path: str = "core/agents/example.py") -> Dict[str, Any]:
        """
        Statically checks a code snippet against AFOS v30.1 Path A architectural invariants:
        - UI isolation (no streamlit/react in core/ or adam_*/)
        - Path A/B bifurcation (no experimental/ imports in core/)
        - P0 Security mandates (no pickle.load, eval, exec)
        """
        violations: List[str] = []
        is_path_a = module_path.startswith(("core/", "adam_", "src/"))

        if is_path_a:
            for ui_pkg in self.FORBIDDEN_CORE_UI_IMPORTS:
                pattern = rf"^\s*(import\s+{ui_pkg}\b|from\s+{ui_pkg}\b)"
                if re.search(pattern, code, flags=re.MULTILINE):
                    violations.append(f"UI_ISOLATION_BREACH: '{ui_pkg}' imported in Path A module '{module_path}'")

            for lab_prefix in self.FORBIDDEN_LAB_PREFIXES:
                base_pkg = lab_prefix.rstrip(".")
                pattern = rf"^\s*(import\s+{base_pkg}\b|from\s+{base_pkg}\b)"
                if re.search(pattern, code, flags=re.MULTILINE):
                    violations.append(
                        f"BIFURCATION_BREACH: Path B module '{base_pkg}' imported in Path A module '{module_path}'"
                    )

        for p0_call in self.FORBIDDEN_P0_CALLS:
            if p0_call in code:
                violations.append(f"P0_SECURITY_BREACH: Forbidden pattern '{p0_call}' detected in '{module_path}'")

        frontend_decoupled = not any(v.startswith("UI_ISOLATION_BREACH") for v in violations)
        self.state.frontend_sync_status = "decoupled" if frontend_decoupled else "violation_detected"

        return {
            "module_path": module_path,
            "is_path_a": is_path_a,
            "passed": len(violations) == 0,
            "violations": violations,
            "frontend_sync_status": self.state.frontend_sync_status,
        }

    def prepare_jules_session(
        self,
        request: JulesSessionRequest,
        execute: bool = False,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Prepares (and optionally executes) a `@google/jules` CLI command with full
        W3C PROV-O telemetry and state tracking.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        args = JulesCommandBuilder.build_args(request)
        cli_string = JulesCommandBuilder.format_cli_string(request)
        jules_installed = shutil.which(JulesCommandBuilder.BINARY) is not None

        execution_result: Dict[str, Any] = {
            "executed": False,
            "jules_binary_detected": jules_installed,
            "returncode": None,
            "stdout": None,
            "stderr": None,
        }

        if execute:
            if not jules_installed:
                execution_result["stderr"] = (
                    "Jules CLI ('jules') not found on PATH. Install globally via: npm install -g @google/jules"
                )
                execution_result["returncode"] = 127
            else:
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                execution_result.update(
                    {
                        "executed": True,
                        "returncode": proc.returncode,
                        "stdout": proc.stdout.strip(),
                        "stderr": proc.stderr.strip(),
                    }
                )

        payload = {
            "agent_role": self.state.agent_role,
            "active_bounded_context": self.state.active_bounded_context,
            "frontend_sync_status": self.state.frontend_sync_status,
            "subcommand": request.subcommand,
            "command_args": args,
            "command_string": cli_string,
            "request": request.model_dump(),
            "execution": execution_result,
        }

        canonical_json = json.dumps(payload, sort_keys=True)
        content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        confidence = 0.95 if self.state.frontend_sync_status == "decoupled" else 0.40

        provenance = ProvenanceHeader(
            git_commit_hash=self.state.last_checkpoint_hash,
            timestamp=now_utc,
            content_hash=content_hash,
            jsonLogic_version="2.0.0",
            confidence_score=confidence,
            derivation_path=f"architect_jules -> jules_cli:{request.subcommand}",
            source_data_object="urn:adam:agent:architect_jules:v30.1",
        )

        prov_entry = provenance.model_dump()
        self.state.prov_o_audit_trail.append(prov_entry)

        return {
            "answer": f"Prepared Jules CLI command: `{cli_string}`",
            "sources": ["docs/jules_tools_reference.md", "AGENTS.md", "AGENT_CATALOG.md"],
            "confidence": confidence,
            "metadata": payload,
            "prov_o_audit_trail": list(self.state.prov_o_audit_trail),
            "provenance_trace": prov_entry,
        }


__all__ = [
    "ArchitectAgent",
    "ArchitectJulesState",
    "JulesCommandBuilder",
    "JulesSessionRequest",
]

