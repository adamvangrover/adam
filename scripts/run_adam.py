import sys
import os
import asyncio
import logging
import argparse

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.config_utils import load_app_config
from core.utils.logging_utils import setup_logging
from core.engine.meta_orchestrator import MetaOrchestrator
from core.system.agent_orchestrator import AgentOrchestrator

async def main():
    """
    Main execution script for Adam v23.0 (Adaptive System).
    """
    parser = argparse.ArgumentParser(description="Adam v23.0 Execution")
    parser.add_argument("--query", type=str, help="Single query to execute")
    args = parser.parse_args()

    try:
        # Load configuration
        # Note: load_app_config might need to be robust if config files are missing
        try:
            config = load_app_config()
        except Exception as e:
            print(f"Warning: Could not load full config ({e}). Using defaults.")
            config = {}

        # Set up logging
        try:
            setup_logging(config)
        except Exception:
            logging.basicConfig(level=logging.INFO)

        logger = logging.getLogger(__name__)
        logger.info("Initializing Adam v23.0 System...")

        # Initialize Legacy Orchestrator (v21/v22)
        try:
            legacy_orchestrator = AgentOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {e}")
            legacy_orchestrator = None # Fallback

        # Initialize Meta Orchestrator (v23 Brain)
        meta_orchestrator = MetaOrchestrator(legacy_orchestrator=legacy_orchestrator)

        if args.query:
            # Single shot mode
            print(f"User> {args.query}")
            print("Adam> Thinking...")
            result = await meta_orchestrator.route_request(args.query)
            print(f"Adam> {result}")
            return

        print("\n" + "="*50)
        print("   ADAM v23.0 - Adaptive Financial Intelligence")
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
                result = await meta_orchestrator.route_request(user_input)

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
        # sys.exit(1) # Don't exit with error code if it's just a runtime issue, let's keep it clean

if __name__ == "__main__":
    asyncio.run(main())
