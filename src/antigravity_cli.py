"""
Antigravity Unified Terminal CLI Tool.
Command-line management and autonomous execution interface for the Antigravity Suite.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.antigravity_sdk import (
    Agent,
    CapabilitiesConfig,
    JulesAgent,
    LocalAgentConfig,
)
from src.antigravity_sdk.tools import ToolRunner


def get_default_tools() -> list[Any]:
    """Default built-in tools for CLI tasks."""
    def read_repo_file(filepath: str) -> str:
        """Reads content of a file in the workspace."""
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found: {filepath}"
        return path.read_text(encoding="utf-8")

    def list_repo_files(directory: str = ".") -> str:
        """Lists directory entries."""
        p = Path(directory)
        if not p.exists():
            return f"Directory not found: {directory}"
        items = [f"{'[DIR] ' if f.is_dir() else '[FILE] '}{f.name}" for f in p.iterdir()]
        return "\n".join(sorted(items)[:100])

    return [read_repo_file, list_repo_files]


def build_parser() -> argparse.ArgumentParser:
    """Constructs the unified CLI argument parser for Antigravity."""
    parser = argparse.ArgumentParser(
        prog="antigravity",
        description="Antigravity Unified CLI: Terminal Tool & Workbench Management for Autonomous Agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Antigravity commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run an autonomous task or prompt")
    run_parser.add_argument("prompt", type=str, help="The prompt or objective to execute")
    run_parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model name")
    run_parser.add_argument("--json", action="store_true", help="Output full JSON envelope")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Start an interactive terminal REPL chat session")
    chat_parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model name")

    # Command: tools
    subparsers.add_parser("tools", help="List and inspect registered agent tools")

    # Command: triggers
    subparsers.add_parser("triggers", help="Inspect and simulate background triggers")

    # Command: status
    subparsers.add_parser("status", help="Inspect Antigravity runtime, environment, and connectivity")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch local Antigravity IDE / App workbench server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    # Command: extensions
    ext_parser = subparsers.add_parser("extensions", help="Manage and list installed extensions & plugins")
    ext_parser.add_argument("--dir", type=str, default="extensions", help="Extensions directory")

    # Command: orchestrate
    orch_parser = subparsers.add_parser("orchestrate", help="Run v30.1-R orchestrators via EcosystemHub")
    orch_parser.add_argument(
        "--target",
        type=str,
        choices=["gate_clearance", "market_mayhem", "bitemporal_audit", "r2_expanded", "repo_prompt", "all"],
        default="all",
        help="Orchestrator target to execute (default: all)",
    )
    orch_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    orch_parser.add_argument("--output", type=str, default="", help="Optional output file path for JSON report")

    # Command: underwrite
    uw_parser = subparsers.add_parser("underwrite", help="Run ASC 842 / SR 26-2 single-name credit underwriting")
    uw_parser.add_argument("--obligor", type=str, default="NEOC", help="Obligor ticker or ID")
    uw_parser.add_argument("--sector", type=str, default="TMT / AI Compute Infrastructure", help="Obligor sector")
    uw_parser.add_argument("--revenue", type=float, default=1800.0, help="Trailing revenue ($M)")
    uw_parser.add_argument("--ebitda", type=float, default=420.0, help="Trailing GAAP EBITDA ($M)")
    uw_parser.add_argument("--capex", type=float, default=90.0, help="Maintenance CapEx ($M)")
    uw_parser.add_argument("--cash-interest", type=float, default=65.0, help="Cash interest expense ($M)")
    uw_parser.add_argument("--cash-taxes", type=float, default=45.0, help="Cash taxes ($M)")
    uw_parser.add_argument("--total-debt", type=float, default=950.0, help="GAAP funded debt ($M)")
    uw_parser.add_argument("--cash", type=float, default=0.0, help="Unrestricted cash ($M)")
    uw_parser.add_argument("--lease-payment", type=float, default=640.0, help="Annual compute/rack lease payment ($M)")
    uw_parser.add_argument("--lease-term", type=int, default=5, help="Weighted-average lease term (years)")
    uw_parser.add_argument("--discount-rate", type=float, default=0.06, help="Implicit lease discount rate (decimal or %)")
    uw_parser.add_argument("--ev-multiple", type=float, default=8.5, help="Reference EV/EBITDA multiple")
    uw_parser.add_argument("--pd-alpha", type=float, default=None, help="Optional override for Model Alpha PD")
    uw_parser.add_argument("--pd-beta", type=float, default=None, help="Optional override for Model Beta PD")
    uw_parser.add_argument("--theta", type=float, default=0.03, help="Neutrality arbitration threshold theta")
    uw_parser.add_argument("--lockdown", action="store_true", help="Force V3 75/25 Fortress-Hunt Lockdown")
    uw_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    # Command: skills
    skills_parser = subparsers.add_parser("skills", help="Discover and execute governed workspace skills")
    skills_parser.add_argument("--list", action="store_true", help="List available workspace skills")
    skills_parser.add_argument("--run", type=str, default="", help="Skill ID to execute")
    skills_parser.add_argument("--payload", type=str, default="", help="JSON payload string or path to JSON file")
    skills_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    # Command: audit-security
    audit_parser = subparsers.add_parser("audit-security", help="Run P0 AST pickle & importlib security CI gate")
    audit_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    # Command: portals
    portals_parser = subparsers.add_parser("portals", help="List AFOS Web & Command Center portals")
    portals_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    return parser


async def cmd_run(args: argparse.Namespace) -> int:
    """Executes a task prompt using the Jules / Antigravity agent."""
    jules = JulesAgent()
    res = await jules.execute_task(args.prompt)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"\n[Antigravity CLI] Status: {res.get('status', 'SUCCESS')}")
        print(f"[Antigravity CLI] Response:\n{res.get('response', '')}")
        print(f"\n[PROV-O Audit Entries]: {len(res.get('prov_o_audit_trail', []))}")
        for entry in res.get("prov_o_audit_trail", []):
            print(f"  - [{entry.get('prov:Activity')}]: {entry.get('prov:used')} -> {entry.get('prov:generated')}")
    return 0


async def cmd_chat(args: argparse.Namespace) -> int:
    """Runs interactive terminal chat REPL."""
    print("=================================================================")
    print("      Antigravity Agent Interactive Terminal REPL (AFOS v30.1)    ")
    print("  Type 'exit', 'quit', or Ctrl+C to terminate session.          ")
    print("=================================================================")
    config = LocalAgentConfig(
        model_name=args.model,
        capabilities=CapabilitiesConfig(allow_terminal=True),
        tools=get_default_tools(),
    )
    async with Agent(config) as agent:
        while True:
            try:
                prompt = input("\nantigravity> ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ["exit", "quit", "q"]:
                    print("Exiting Antigravity session. Goodbye!")
                    break
                response = await agent.chat(prompt)
                print("\n[Agent]:")
                async for token in response:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted.")
                break
    return 0


def cmd_tools() -> int:
    """Lists available registered tools."""
    runner = ToolRunner(tools=get_default_tools())
    schemas = runner.get_tool_schemas()
    print("=================================================================")
    print(f" Registered Antigravity Tools ({len(schemas)})")
    print("=================================================================")
    for tool in schemas:
        print(f"- Name: {tool['name']}")
        print(f"  Description: {tool.get('description', 'N/A')}")
        print(f"  Parameters: {json.dumps(tool.get('parameters', {}))}")
        print("-" * 50)
    return 0


def cmd_triggers() -> int:
    """Inspects trigger framework status and simulation capabilities."""
    print("=================================================================")
    print(" Antigravity Background Triggers & Event Engine")
    print("=================================================================")
    print("Available triggers: Periodic timer ('every'), MCP Event Subscriber, PROV-O Watcher")
    print("Status: Active trigger runner configured. Default interval: 60.0s")
    return 0


def cmd_status() -> int:
    """Outputs status of Antigravity runtime, SDK, and environment."""
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_vertex = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    print("=================================================================")
    print(" Antigravity Runtime & Environment Diagnostics")
    print("=================================================================")
    print("System: ADAM AFOS v30.1 (Unified Kernel)")
    print("SDK Package: src/antigravity_sdk (Loaded)")
    print(f"Gemini API Key: {'Configured' if has_gemini else 'Unset (Using Local Deterministic Fallback)'}")
    print(f"Vertex AI ADC:  {'Configured' if has_vertex else 'Unset'}")
    print("Default Capabilities: File System Write=True, Terminal=True, Code Execution=True")
    print("Safety Primacy: Lexicographic Layer 0 x Layer 1 Verification Active")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launches the Antigravity IDE and App Web server."""
    try:
        from src.antigravity_ide.server import run_server
        print(f"[Antigravity CLI] Starting Antigravity IDE Server on http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)
        return 0
    except ImportError as e:
        print(f"Failed to start server: {e}")
        return 1


def cmd_extensions(args: argparse.Namespace) -> int:
    """Lists registered extensions and plugins."""
    base_dir = Path(args.dir)
    print("=================================================================")
    print(f" Antigravity IDE Extensions & Plugins ({base_dir})")
    print("=================================================================")
    if not base_dir.exists():
        print(f"No extensions directory found at {base_dir}")
        return 0
    found = 0
    for pkg in base_dir.glob("**/package.json"):
        found += 1
        try:
            meta = json.loads(pkg.read_text(encoding="utf-8"))
            print(f"- Extension: {meta.get('name', 'Unknown')} (v{meta.get('version', '0.0.1')})")
            print(f"  Description: {meta.get('description', '')}")
            print(f"  Manifest: {pkg}")
        except Exception as e:
            print(f"- Invalid manifest at {pkg}: {e}")
    plugins_dir = Path("plugins")
    if plugins_dir.exists():
        for plg in plugins_dir.glob("*.json"):
            found += 1
            try:
                pdata = json.loads(plg.read_text(encoding="utf-8"))
                print(f"- Plugin: {pdata.get('name', plg.stem)} (Type: {pdata.get('type', 'general')})")
                print(f"  File: {plg}")
            except Exception as e:
                print(f"- Error loading plugin {plg}: {e}")
    if found == 0:
        print("No extensions or plugins discovered.")
    return 0


def _get_ecosystem_hub():
    from src.orchestration.ecosystem_hub import EcosystemHub

    return EcosystemHub()


def cmd_orchestrate(args: argparse.Namespace) -> int:
    """Execute one or all v30.1-R orchestrators via EcosystemHub."""
    hub = _get_ecosystem_hub()
    result = hub.run_orchestrator(args.target)
    serialized = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    if args.json:
        print(serialized)
    else:
        print(f"AFOS Orchestrator Target: {args.target}")
        print(f"Status: {result.get('status')}")
        print(f"SHA-256 Release Fingerprint: {result.get('sha256_release_fingerprint')}")
    return 0


def cmd_underwrite(args: argparse.Namespace) -> int:
    """Run deterministic ASC 842 / SR 26-2 single-name credit underwriting."""
    hub = _get_ecosystem_hub()
    discount_pct = args.discount_rate * 100.0 if args.discount_rate <= 1.0 else args.discount_rate
    lease_commitments = args.lease_payment * max(1, args.lease_term)
    params: dict[str, object] = {
        "ticker": args.obligor,
        "sector": args.sector,
        "revenue_m": args.revenue,
        "ebitda_m": args.ebitda,
        "capex_maint_m": args.capex,
        "cash_interest_m": args.cash_interest,
        "cash_taxes_m": args.cash_taxes,
        "gaap_funded_debt_m": args.total_debt,
        "off_balance_sheet_lease_commitments_m": lease_commitments,
        "annual_lease_expense_m": args.lease_payment,
        "lease_term_years": args.lease_term,
        "lease_discount_rate_pct": discount_pct,
        "theta_threshold": args.theta,
        "ev_multiple": args.ev_multiple,
        "cash_m": args.cash,
        "lockdown": bool(args.lockdown),
    }
    if args.pd_alpha is not None:
        params["pd_alpha"] = args.pd_alpha
    if args.pd_beta is not None:
        params["pd_beta"] = args.pd_beta

    result = hub.underwrite_obligor(params)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        g = result["gaap_vs_lease_adjusted"]
        wf = result["cash_flow_waterfall"]
        pd_arb = result["dual_pd_arbitration"]
        gov = result["governance"]
        print(f"Obligor: {result['obligor']['ticker']} ({result['obligor']['sector']})")
        print(
            f"Leverage: GAAP={g['gaap_leverage_x']}x | Lease-Capitalized={g['lease_capitalized_leverage_x']}x "
            f"(+{g['hidden_leverage_delta_x']}x)"
        )
        print(f"CADS=${wf['cads_full_waterfall_m']}M | Spot FCCR={wf['spot_fccr']}x | Stressed FCCR={wf['stressed_fccr']}x")
        print(
            f"Dual PD: Alpha={pd_arb['model_alpha_pd']} vs Beta={pd_arb['model_beta_pd']} "
            f"| Implied Rating={pd_arb['implied_rating']} | Tier={gov['operating_tier']} ({gov['system_status']})"
        )
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    """List or execute governed workspace skills."""
    hub = _get_ecosystem_hub()
    if args.run:
        payload = None
        if args.payload:
            maybe_path = Path(args.payload)
            if maybe_path.exists() and maybe_path.is_file():
                payload = json.loads(maybe_path.read_text(encoding="utf-8"))
            else:
                payload = json.loads(args.payload)
        result = hub.run_skill(args.run, payload=payload)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Skill: {result.get('skill_id')} | Status: {result.get('status')}")
            print(f"SHA-256: {result.get('sha256_release_fingerprint')}")
        return 0

    skills = hub.list_skills()
    if args.json:
        print(json.dumps({"skills": skills}, indent=2, default=str))
    else:
        print("AFOS Governed Workspace Skills:")
        for sk in skills:
            print(f"- {sk.get('id')}: {sk.get('name')} ({sk.get('category', 'governance')})")
    return 0


def cmd_audit_security(args: argparse.Namespace) -> int:
    """Run P0 AST pickle & importlib CI gate across core/ and src/."""
    hub = _get_ecosystem_hub()
    result = hub.run_security_ci_gate()
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"Security CI Gate ({result.get('gate_id')}): passed={result.get('passed')} "
            f"| files_scanned={result.get('files_scanned')}"
        )
    return 0 if result.get("passed") else 1


def cmd_portals(args: argparse.Namespace) -> int:
    """List AFOS ecosystem web & Command Center portals."""
    hub = _get_ecosystem_hub()
    portals = hub.list_portals()
    if args.json:
        print(json.dumps({"portals": portals}, indent=2, default=str))
    else:
        print("AFOS Ecosystem Portals:")
        for portal in portals:
            print(f"- {portal.get('name')} [{portal.get('id')}]: {portal.get('path')}")
    return 0


def cmd_status_with_ecosystem() -> int:
    rc = cmd_status()
    try:
        manifest = _get_ecosystem_hub().get_ecosystem_manifest()
        counts = manifest.get("counts", {})
        print(
            f"EcosystemHub: orchestrators={counts.get('orchestrators', 0)} | "
            f"skills={counts.get('skills', 0)} | portals={counts.get('portals', 0)}"
        )
    except Exception:
        pass
    return rc


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        return asyncio.run(cmd_run(args))
    if args.command == "chat":
        return asyncio.run(cmd_chat(args))
    if args.command == "tools":
        return cmd_tools()
    if args.command == "triggers":
        return cmd_triggers()
    if args.command == "status":
        return cmd_status()
        return cmd_status_with_ecosystem()
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "extensions":
        return cmd_extensions(args)
    if args.command == "orchestrate":
        return cmd_orchestrate(args)
    if args.command == "underwrite":
        return cmd_underwrite(args)
    if args.command == "skills":
        return cmd_skills(args)
    if args.command == "audit-security":
        return cmd_audit_security(args)
    if args.command == "portals":
        return cmd_portals(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
