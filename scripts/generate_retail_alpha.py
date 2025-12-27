import asyncio
import json
import os
import sys

# Ensure the project root is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.agents.specialized.retail_alpha_agent import RetailAlphaAgent

async def main():
    print("Initializing Retail Alpha Agent...")

    # Config can be minimal for this run
    config = {
        "mock_mode": True,
        "log_level": "INFO"
    }

    agent = RetailAlphaAgent(config)

    # Defined watchlist suitable for retail traders
    watchlist = [
        "NVDA", "TSLA", "AAPL", "AMD", "PLTR",
        "GME", "COIN", "MARA", "MSTR", "HOOD",
        "SOFI", "PATH", "UPST", "AI"
    ]

    print(f"Running analysis on {len(watchlist)} tickers...")
    data = await agent.execute(tickers=watchlist)

    # Ensure output directory exists
    output_dir = os.path.join("showcase", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "retail_alpha.json")

    print(f"Saving report to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Done! Retail Alpha data generated.")

if __name__ == "__main__":
    asyncio.run(main())
