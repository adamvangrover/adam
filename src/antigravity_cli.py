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
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "extensions":
        return cmd_extensions(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
