#!/usr/bin/env python3
"""
AFOS v30.1 Architect Agent (Jules) Workflow Runner
==================================================
Orchestrates `@google/jules` CLI coding sessions mapped to AFOS v30.1 architectural
priorities in `docs/AGENTS_KNOWLEDGE_BASE.md` and `AGENTS.md`, emitting W3C PROV-O
provenance bundles.

Usage:
  uv run python scripts/run_jules_workflows.py --list-presets
  uv run python scripts/run_jules_workflows.py --preset p0_security_hardening --parallel 2
  uv run python scripts/run_jules_workflows.py --repo adamvangrover/adam --session "write unit tests"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.architect_agent import (
    ArchitectAgent,
    JulesSessionRequest,
)

AFOS_JULES_PRESETS: dict[str, dict[str, Any]] = {
    "p0_security_hardening": {
        "bounded_context": "v30.1_p0_security_hardening",
        "repo": "adamvangrover/adam",
        "session_prompt": (
            "Audit core/ and adam_*/ for P0 security mandates per docs/AGENTS_KNOWLEDGE_BASE.md: "
            "replace any pickle.load() with core.security.safe_unpickler.safe_load(), eliminate "
            "unwhitelisted importlib dynamic imports, and add unit tests verifying adversarial inputs."
        ),
        "default_parallel": 2,
    },
    "p1_graph_consolidation": {
        "bounded_context": "v30.1_graph_engine_consolidation",
        "repo": "adamvangrover/adam",
        "session_prompt": (
            "Consolidate duplicate UnifiedKnowledgeGraph implementations in core/engine/ and "
            "enforce singleton caching and W3C PROV-O ProvenanceHeader emission across System 2 DAG nodes."
        ),
        "default_parallel": 1,
    },
    "p1_pdil_rust_migration": {
        "bounded_context": "v30.1_pdil_gatekeeper_optimization",
        "repo": "adamvangrover/adam",
        "session_prompt": (
            "Optimize PDIL (Probabilistic-to-Deterministic Integration Layer) schema validation "
            "and W3C PROV-O SHA-256 verification across src/pdil/ and rust_ext/ with zero-drift tests."
        ),
        "default_parallel": 2,
    },
    "p2_frontend_isolation_audit": {
        "bounded_context": "v30.1_frontend_decoupling",
        "repo": "adamvangrover/adam",
        "session_prompt": (
            "Verify strict Path A isolation so no UI frameworks (streamlit, react) or Path B (experimental/) "
            "modules are imported inside core/ or adam_*/, and audit aria-live accessibility attributes in showcase/."
        ),
        "default_parallel": 1,
    },
    "p0_remote_status_audit": {
        "bounded_context": "v30.1_remote_status_telemetry",
        "repo": "adamvangrover/adam",
        "session_prompt": (
            "Query active remote sessions status and construct W3C PROV-O audit bundle for session tracking."
        ),
        "default_parallel": 1,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AFOS v30.1 Architect Agent (Jules) CLI Workflow Bridge"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List predefined AFOS v30.1 Jules architectural workflow presets.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(AFOS_JULES_PRESETS.keys()),
        default=None,
        help="Predefined AFOS v30.1 architectural preset to prepare/dispatch.",
    )
    parser.add_argument(
        "--subcommand",
        choices=[
            "version",
            "help",
            "login",
            "logout",
            "tui",
            "remote_list_repo",
            "remote_list_session",
            "remote_new",
            "remote_pull",
            "remote_status",
            "completion",
        ],
        default="remote_new",
        help="Jules CLI subcommand to invoke.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="adamvangrover/adam",
        help="Repository identifier (e.g. 'adamvangrover/adam', 'torvalds/linux', or '.').",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Prompt for `remote_new` or session ID for `remote_pull`.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel sessions to launch (`--parallel <number>`).",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default=None,
        help="Optional TUI theme (`dark` or `light`).",
    )
    parser.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish", "powershell"],
        default="zsh",
        help="Target shell when `--subcommand completion` is used.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the `jules` CLI binary directly instead of dry-run command generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write the JSON W3C PROV-O output bundle.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_presets:
        print(json.dumps(AFOS_JULES_PRESETS, indent=2))
        return

    bounded_context = "v30.1_async_migration"
    repo = args.repo
    session_prompt = args.session
    session_id = args.session if args.subcommand == "remote_pull" else None
    parallel = args.parallel

    if args.preset:
        preset_cfg = AFOS_JULES_PRESETS[args.preset]
        bounded_context = preset_cfg["bounded_context"]
        repo = preset_cfg["repo"]
        session_prompt = preset_cfg["session_prompt"]
        if parallel is None:
            parallel = preset_cfg["default_parallel"]

    if args.subcommand == "remote_new" and not session_prompt:
        session_prompt = AFOS_JULES_PRESETS["p0_security_hardening"]["session_prompt"]

    request = JulesSessionRequest(
        subcommand=args.subcommand,
        repo=repo if args.subcommand == "remote_new" else None,
        session_prompt=session_prompt if args.subcommand == "remote_new" else None,
        session_id=session_id,
        parallel=parallel if args.subcommand == "remote_new" else None,
        theme=args.theme,
        shell=args.shell,
    )

    agent = ArchitectAgent(active_bounded_context=bounded_context)
    result = agent.prepare_jules_session(request=request, execute=args.execute)

    formatted = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(formatted, encoding="utf-8")
        print(f"Jules workflow report written to {args.output}")
    else:
        print(formatted)


if __name__ == "__main__":
    main()

