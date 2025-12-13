import sys
import os
import asyncio
import logging
import argparse
from core.utils.config_utils import load_app_config
from core.utils.logging_utils import setup_logging
from core.engine.meta_orchestrator import MetaOrchestrator
from core.system.agent_orchestrator import AgentOrchestrator
from core.settings import settings

async def async_main():
    """
    Main execution logic for Adam v23.0 (Adaptive System).
    """
    parser = argparse.ArgumentParser(description="Adam v23.0 Execution")
    parser.add_argument("--query", type=str, help="Single query to execute")
    parser.add_argument("--system_prompt", type=str, help="System Prompt to inject (String)")
    parser.add_argument("--system_prompt_path", type=str, help="System Prompt to inject (File Path)")
    args = parser.parse_args()

    try:
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

        # Initialize Legacy Orchestrator (v21/v22)
        try:
            legacy_orchestrator = AgentOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {e}")
            legacy_orchestrator = None # Fallback

        # Initialize Meta Orchestrator (v23 Brain)
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

def main():
    """Synchronous entry point for console_scripts."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
