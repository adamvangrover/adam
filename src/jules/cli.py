"""
Jules Helper CLI Interface.
Command-line interface for autonomous task execution, planning, REPL chat, and web server execution.
"""

import argparse
import json
import sys

from src.jules.sdk.agent import JulesAgent, JulesEngine
from src.jules.sdk.config import JulesConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jules",
        description="Jules Helper CLI: Self-contained AI Assistant & Automation Engine for ADAM OS",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Jules commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run an autonomous task prompt")
    run_parser.add_argument("prompt", type=str, help="The task prompt for Jules to execute")
    run_parser.add_argument("--provider", type=str, default="mock", help="LLM provider (mock, ollama, openai, anthropic)")
    run_parser.add_argument("--model", type=str, default="mock-jules-v1", help="Model identifier")

    # Command: plan
    plan_parser = subparsers.add_parser("plan", help="Generate a step-by-step execution plan for a prompt")
    plan_parser.add_argument("prompt", type=str, help="The task prompt to formulate a plan for")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Start an interactive REPL chat session with Jules")
    chat_parser.add_argument("--provider", type=str, default="mock", help="LLM provider")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch local Jules IDE Web Server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port number")

    # Command: info
    info_parser = subparsers.add_parser("info", help="Display Jules environment status, active provider, and registered tools")

    return parser


def handle_run(args: argparse.Namespace) -> int:
    agent = JulesEngine.create(provider=args.provider, model_name=args.model)
    print(f"\n🚀 [Jules CLI] Executing task: '{args.prompt}'\n")
    result = agent.run(args.prompt)
    print("\n📋 Execution Plan:")
    for step in result["plan"]:
        print(f"  {step}")
    print(f"\nStatus: {result['status']}")
    print(f"Audit log entries: {len(result['audit_log'])}\n")
    return 0


def handle_plan(args: argparse.Namespace) -> int:
    agent = JulesEngine.create()
    plan = agent.set_plan(args.prompt)
    print(f"\n📋 Execution Plan for: '{args.prompt}'\n")
    for step in plan:
        print(f"  {step}")
    print()
    return 0


def handle_chat(args: argparse.Namespace) -> int:
    agent = JulesEngine.create(provider=args.provider)
    print("\n💬 [Jules Chat REPL] Type 'exit' or 'quit' to end session.\n")
    while True:
        try:
            user_input = input("jules> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            reply = agent.chat(user_input)
            print(f"\nJules: {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    try:
        from src.jules.ide.server import run_server
        print(f"Starting Jules IDE Server at http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)
        return 0
    except ImportError:
        print("Error: Jules IDE server dependencies not available.")
        return 1


def handle_info(args: argparse.Namespace) -> int:
    cfg = JulesConfig()
    agent = JulesAgent(config=cfg)
    tools = agent.tool_registry.list_tools()
    history = agent.memory.get_history(limit=5)

    info_data = {
        "jules_version": "1.0.0",
        "provider": cfg.provider,
        "model": cfg.model_name,
        "workspace_root": cfg.workspace_root,
        "registered_tools_count": len(tools),
        "tools": [t["name"] for t in tools],
        "recent_task_history_count": len(history),
    }

    print("\nℹ️  [Jules System Status & Environment]")
    print(json.dumps(info_data, indent=2))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        return handle_run(args)
    elif args.command == "plan":
        return handle_plan(args)
    elif args.command == "chat":
        return handle_chat(args)
    elif args.command == "serve":
        return handle_serve(args)
    elif args.command == "info":
        return handle_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
