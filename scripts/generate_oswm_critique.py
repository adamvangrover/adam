import asyncio
import json
import logging
import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.strategic_foresight_agent import StrategicForesightAgent
from core.utils.logging_utils import setup_logging

async def main():
    setup_logging()
    logging.info("Starting OSWM Critique Generation...")

    config = {
        "name": "StrategicForesightAgent",
        "log_path": "showcase/data/unified_banking_log.json"
    }

    agent = StrategicForesightAgent(config)

    try:
        result = await agent.execute()

        output_path = "showcase/data/oswm_insight.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logging.info(f"OSWM Critique saved to {output_path}")
        print(f"SUCCESS: Generated {output_path}")

    except Exception as e:
        logging.error(f"Failed to generate critique: {e}")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
