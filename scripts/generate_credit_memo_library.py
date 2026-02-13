import json
import asyncio
import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipelines.mock_edgar import MockEdgar
from core.pipelines.credit_pipeline import CreditPipeline

PORTFOLIO = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corp", "sector": "Technology"},
    {"ticker": "NVDA", "name": "NVIDIA Corp", "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer"},
    {"ticker": "META", "name": "Meta Platforms", "sector": "Technology"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financial"},
    {"ticker": "GS", "name": "Goldman Sachs", "sector": "Financial"},
    {"ticker": "BAC", "name": "Bank of America", "sector": "Financial"}
]

async def generate_library():
    print("Starting Credit Memo Library Generation (Full Run)...")

    pipeline = CreditPipeline()
    library = {}

    for entity in PORTFOLIO:
        print(f"Processing {entity['ticker']}...")
        try:
            # 1. Generate Raw Data via MockEdgar
            # Note: We use the pipeline logic to ensure consistency
            # But the pipeline returns the *Memo*. We also want the *Source Data* for the library.
            # The pipeline.run returns {"memo": ..., "source_data": ...}

            result = await pipeline.run(entity["ticker"], entity["name"], entity["sector"])

            # 2. Structure for Frontend
            # We need to match the structure expected by credit_memo_v2.js
            # which is dict of Key -> {borrower_details, documents, market_data}
            # The pipeline.run returns "source_data" which has exactly this.

            library[entity["name"]] = result["source_data"]

            # 3. Inject Generated Memo into the library?
            # The current JS simulates the writer.
            # Ideally, the "Pre-generated" library should include the AI output so the JS doesn't have to re-simulate it if we want "Static" mode.
            # But the current JS *is* a simulator. It uses the raw data to *show* the process.
            # So storing the raw data is sufficient for the "Simulated" fallback.

        except Exception as e:
            print(f"Error processing {entity['ticker']}: {e}")
            import traceback
            traceback.print_exc()

    # Save
    out_path = "showcase/data/credit_memo_library.json"
    with open(out_path, 'w') as f:
        json.dump(library, f, indent=2)

    print(f"Library generated at {out_path} with {len(library)} entities.")

if __name__ == "__main__":
    asyncio.run(generate_library())
