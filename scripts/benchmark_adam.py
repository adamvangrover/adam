import time
import sys
import os
import subprocess
import logging

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.utils.logging_utils import SwarmLogger

def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def run_benchmark():
    logger = SwarmLogger()
    git_hash = get_git_revision_hash()

    print(f"Starting Adam Benchmark (Commit: {git_hash})...")
    logger.log_event("BENCHMARK_START", "BenchmarkRunner", {"git_hash": git_hash})

    # Setup Benchmark - UKG Ingestion
    try:
        ukg_bench = UnifiedKnowledgeGraph()
        n_covenants_bench = 5000 # Moderate load for CI/Dev
        risk_state_bench = {
            "ticker": "BENCH_TEST",
            "balance_sheet": {"fiscal_year": "2023", "total_debt": 1000},
            "income_statement": {"consolidated_ebitda": 500},
            "covenants": [
                {"name": f"Cov_{i}", "threshold": 4.5, "operator": "<="}
                for i in range(n_covenants_bench)
            ],
            "draft_memo": {"recommendation": "Buy", "confidence_score": 0.9}
        }

        print(f"Benchmarking UKG Ingestion ({n_covenants_bench} covenants)...")
        start_time = time.time()
        ukg_bench.ingest_risk_state(risk_state_bench)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Time taken: {duration:.4f} seconds")

        # Log Result
        logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
            "git_hash": git_hash,
            "benchmark_name": "UKG_Ingestion",
            "iterations": n_covenants_bench,
            "duration_seconds": duration,
            "status": "SUCCESS"
        })

    except Exception as e:
        print(f"Benchmark Failed: {e}")
        logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
            "git_hash": git_hash,
            "benchmark_name": "UKG_Ingestion",
            "status": "FAILED",
            "error": str(e)
        })
        sys.exit(1)

if __name__ == "__main__":
    # Configure basic logging to stdout for immediate feedback
    logging.basicConfig(level=logging.INFO)
    run_benchmark()
