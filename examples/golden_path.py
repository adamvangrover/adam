import sys
import os
import asyncio
import logging

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system.bootstrap import Bootstrap

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("examples.golden_path")

async def run_golden_path():
    logger.info("Starting Golden Path Execution...")

    # 1. Bootstrap
    if not Bootstrap.run():
        logger.error("Bootstrap failed. Exiting.")
        return

    logger.info("✅ Milestone: Bootstrap Complete")

    # 2. Initialize Orchestrator
    try:
        from core.engine.meta_orchestrator import MetaOrchestrator
        # Mock legacy orchestrator for standalone mode to ensure robustness
        # In a real environment, we would initialize AgentOrchestrator here
        meta_orchestrator = MetaOrchestrator(legacy_orchestrator=None)
        logger.info("✅ Milestone: Orchestrator Initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Orchestrator: {e}")
        return

    # 3. Define a safe, synthetic query
    query = "Analyze the risk profile of Cyberdyne Systems given recent regulatory changes."

    # 4. Execute
    logger.info(f"Executing Query: {query}")
    try:
        # Pass a mock context to avoid external dependencies if needed
        # This demonstrates how to inject context for deterministic execution
        context = {"mode": "golden_path", "mock_data": True}

        # Depending on MetaOrchestrator implementation, result might be a dict or string
        result = await meta_orchestrator.route_request(query, context=context)

        logger.info("✅ Milestone: Execution Complete")
        print("\n--- Result ---")
        print(result)
        print("--------------\n")

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_golden_path())
