"""
Execution Script for Sovereign Credit Pipeline
----------------------------------------------
Runs the sovereign credit pipeline for a set of mock tickers (AAPL, MSFT, GOOGL, AMZN, NVDA).
Generates artifacts in showcase/data/sovereign_artifacts/.
"""

import sys
import os

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pipelines.sovereign_pipeline import SovereignPipeline

if __name__ == "__main__":

    # Configuration
    bundle_path = os.path.join("enterprise_bundle", "adam-sovereign-bundle")
    output_dir = os.path.join("showcase", "data", "sovereign_artifacts")

    # Initialize Pipeline
    pipeline = SovereignPipeline(bundle_path, output_dir)

    # Target Universe
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    print(f"Running Sovereign Pipeline for {len(tickers)} companies...")

    for ticker in tickers:
        pipeline.run_pipeline(ticker)

    print(f"Pipeline Execution Complete. Artifacts saved to {output_dir}")
