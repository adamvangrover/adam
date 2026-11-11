"""
Antigravity CLI Interface.
Command-line runner supporting task execution, REPL chat, plans, IDE serve, plugins, extensions, and setups.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .app import AntigravityApp, AppConfig
from .plugins import default_plugin_registry
from .extensions import default_extension_host


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antigravity",
        description="Antigravity Unified CLI: Orchestrator for App, Extensions, SDK, Runtimes, and IDE",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Antigravity commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run an autonomous task with Antigravity")
    run_parser.add_argument("prompt", type=str, help="The prompt or instruction for execution")
    run_parser.add_argument("--runtime", type=str, default="offline", choices=["offline", "local-openai", "gemini", "litert"], help="Runtime mode")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--runtime", type=str, default="offline", help="Runtime mode")

    # Command: plan
    plan_parser = subparsers.add_parser("plan", help="Formulate an execution plan for a prompt")
    plan_parser.add_argument("prompt", type=str, help="Prompt to create plan for")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch Antigravity Web IDE Server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    serve_parser.add_argument("--port", type=int, default=8888, help="Port")

    # Command: plugins
    plugin_parser = subparsers.add_parser("plugins", help="Manage and inspect plugins")
    plugin_parser.add_argument("--list", action="store_true", help="List active plugins")

    # Command: extensions
    ext_parser = subparsers.add_parser("extensions", help="Manage and inspect extensions")
    ext_parser.add_argument("--list", action="store_true", help="List active extensions")

    # Command: status
    subparsers.add_parser("status", help="Display Antigravity system status")

    # Command: setup
    setup_parser = subparsers.add_parser("setup", help="Run environment setup and workspace verification")
    setup_parser.add_argument("--check", action="store_true", help="Check system prerequisites")

    return parser


def main(args: Optional[list[str]] = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command == "status":
        app = AntigravityApp()
        print(json.dumps(app.get_status(), indent=2))
        return 0

    if parsed_args.command == "plugins":
        plugins = default_plugin_registry.list_plugins()
        print(f"Active Plugins ({len(plugins)}):")
        for p in plugins:
            print(f" - {p['name']} v{p['version']}: {p['description']}")
        return 0

    if parsed_args.command == "extensions":
        exts = default_extension_host.list_extensions()
        print(f"Active Extensions ({len(exts)}):")
        for ext in exts:
            print(f" - {ext['name']} ({ext['id']}) v{ext['version']}: {ext['description']}")
        return 0

    if parsed_args.command == "run":
        app = AntigravityApp(AppConfig(runtime_mode=parsed_args.runtime))
        res = app.execute_prompt(parsed_args.prompt)
        print(json.dumps(res, indent=2))
        return 0

    if parsed_args.command == "plan":
        plan = [
            f"1. Deconstruct instruction: {parsed_args.prompt}",
            "2. Consult registered plugins & extensions for domain capabilities",
            "3. Execute neuro-symbolic Layer 0 x Layer 1 safety gate",
            "4. Emit W3C PROV-O audit trail and confirm execution",
        ]
        print("\n".join(plan))
        return 0

    if parsed_args.command == "chat":
        app = AntigravityApp(AppConfig(runtime_mode=parsed_args.runtime))
        print("Starting Antigravity Interactive Console (type 'exit' or 'quit' to end):")
        while True:
            try:
                line = input("antigravity> ")
                if not line.strip():
                    continue
                if line.strip().lower() in ("exit", "quit"):
                    break
                res = app.execute_prompt(line)
                print(res["response"])
            except (EOFError, KeyboardInterrupt):
                print()
                break
        return 0

    if parsed_args.command == "serve":
        from .ide.server import run_server
        print(f"Launching Antigravity Web IDE at http://{parsed_args.host}:{parsed_args.port}")
        run_server(host=parsed_args.host, port=parsed_args.port)
        return 0

    if parsed_args.command == "setup":
        from scripts.setup_antigravity import run_setup
        return run_setup(check_only=parsed_args.check)

    return 0


if __name__ == "__main__":
    sys.exit(main())
