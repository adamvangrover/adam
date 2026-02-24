import sys
import os
import asyncio
import logging
import argparse
from core.utils.config_utils import load_app_config
from core.utils.logging_utils import setup_logging
from core.utils.system_logger import SystemLogger
from core.system.bootstrap import Bootstrap
from core.settings import settings


async def async_main():
    """
    Main execution logic for Adam v26.0 (Neuro-Symbolic Sovereign).

    This function handles:
    1. Parsing command line arguments.
    2. Bootstrapping the environment.
    3. Initializing the Meta Orchestrator.
    4. Running the interactive loop or single-shot query.

    Args:
        None (uses sys.argv via argparse)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Adam v26.0 Execution")
    parser.add_argument("--query", type=str, help="Single query to execute")
    parser.add_argument("--system_prompt", type=str, help="System Prompt to inject (String)")
    parser.add_argument("--system_prompt_path", type=str, help="System Prompt to inject (File Path)")
    parser.add_argument("--legacy", action="store_true", help="Force usage of legacy v23 graph engine components")
    args = parser.parse_args()

    try:
        # Bootstrap Environment
        if not Bootstrap.run():
            print("System Bootstrap Failed. See logs for details.")
            return

        # System Log: Runtime Start
        SystemLogger().log_event("RUNTIME", {"status": "START", "args": sys.argv})

        # Load configuration (Legacy YAML)
        try:
            config = load_app_config()
        except Exception as e:
            print(f"Warning: Could not load full config ({e}). Using defaults.")
            config = {}

        # Set up logging
        try:
            setup_logging(config)
        except Exception:
            logging.basicConfig(level=settings.log_level)

        logger = logging.getLogger("core.main")
        logger.info(f"Initializing {settings.app_name}...")

        # Deferred Imports to allow Bootstrap to run first
        from core.engine.meta_orchestrator import MetaOrchestrator
        from core.system.agent_orchestrator import AgentOrchestrator

        # Initialize Legacy Orchestrator (v21/v22)
        try:
            legacy_orchestrator = AgentOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {e}")
            legacy_orchestrator = None  # Fallback

        # Initialize Meta Orchestrator (v23 Brain)
        # If --legacy is passed, we might modify behavior, but for now we pass it via context or config
        if args.legacy:
            logger.info("LEGACY MODE: Enabled. Routing priority will favor v23_graph_engine components.")

        meta_orchestrator = MetaOrchestrator(legacy_orchestrator=legacy_orchestrator)

        # Inject System Prompt into Context
        context = {}
        if args.system_prompt_path:
            logger.info(f"Injecting System Prompt from {args.system_prompt_path}...")
            try:
                with open(args.system_prompt_path, 'r') as f:
                    context["system_prompt"] = f.read()
            except Exception as e:
                logger.error(f"Failed to read prompt file: {e}")
        elif args.system_prompt:
            logger.info("Injecting System Prompt (String)...")
            context["system_prompt"] = args.system_prompt

        if args.query:
            # Single shot mode
            print(f"User> {args.query}")
            print("Adam> Thinking...")
            result = await meta_orchestrator.route_request(args.query, context=context)
            print(f"Adam> {result}")
            return

        print("\n" + "="*50)
        print(f"   {settings.app_name} - Adaptive Financial Intelligence")
        print("   (Type 'exit' or 'quit' to stop)")
        print("="*50 + "\n")

        # Interactive Loop
        while True:
            try:
                user_input = input("User> ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                # Runtime Seed Injection
                if user_input.startswith("/seed "):
                    new_prompt = user_input[len("/seed "):].strip()
                    if new_prompt:
                        context["system_prompt"] = new_prompt
                        print("Adam> System Prompt Updated (Runtime Seed Injected).")
                        logger.info(f"Runtime Seed Injected: {new_prompt[:50]}...")
                        continue

                if not user_input.strip():
                    continue

                print("Adam> Thinking...")

                # Route request via MetaOrchestrator
                result = await meta_orchestrator.route_request(user_input, context=context)

                print(f"Adam> {result}")

            except EOFError:
                # Handle non-interactive environments
                logger.warning("EOF detected (non-interactive mode). Exiting.")
                break
            except KeyboardInterrupt:
                print("\nAdam> User interrupted.")
                break
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                print(f"Adam> Error: {e}")

    except Exception as e:
        print(f"Fatal Error: {e}")
        SystemLogger().log_event("RUNTIME", {"status": "ERROR", "error": str(e)})
    finally:
        SystemLogger().log_event("RUNTIME", {"status": "STOP"})


def main():
    """Synchronous entry point for console_scripts."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
