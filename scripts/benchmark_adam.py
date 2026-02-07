#!/usr/bin/env python3
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

    ukg_bench = UnifiedKnowledgeGraph()
    n_covenants_bench = 5000 # Moderate load for CI/Dev

    # -------------------------------------------------------------------------
    # Benchmark 1: UKG Ingestion
    # -------------------------------------------------------------------------
    try:
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
        print(f"Ingestion Time: {duration:.4f} seconds")

        logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
            "git_hash": git_hash,
            "benchmark_name": "UKG_Ingestion",
            "iterations": n_covenants_bench,
            "duration_seconds": duration,
            "status": "SUCCESS"
        })

    except Exception as e:
        print(f"Ingestion Benchmark Failed: {e}")
        logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
            "git_hash": git_hash,
            "benchmark_name": "UKG_Ingestion",
            "status": "FAILED",
            "error": str(e)
        })
        # If ingestion fails, query benchmark is moot
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Benchmark 2: UKG Query (Pathfinding)
    # -------------------------------------------------------------------------
    try:
        # Find path from the Entity to the last Covenant
        # Nodes created: LegalEntity::BENCH_TEST -> CreditFacility::... -> Covenant::BENCH_TEST::Cov_...
        start_node = "LegalEntity::BENCH_TEST"
        target_node = f"Covenant::BENCH_TEST::Cov_{n_covenants_bench-1}"

        print(f"Benchmarking UKG Query (Path from {start_node} to {target_node})...")

        start_time = time.time()
        path = ukg_bench.find_symbolic_path(start_node, target_node)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Query Time: {duration:.4f} seconds")

        if not path:
            raise Exception("Path not found (Query returned None)")

        logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
            "git_hash": git_hash,
            "benchmark_name": "UKG_PathQuery",
            "iterations": 1,
            "duration_seconds": duration,
            "status": "SUCCESS"
        })

    except Exception as e:
        print(f"Query Benchmark Failed: {e}")
        logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
            "git_hash": git_hash,
            "benchmark_name": "UKG_PathQuery",
            "status": "FAILED",
            "error": str(e)
        })
        # Don't exit here, just log failure

if __name__ == "__main__":
    # Configure basic logging to stdout for immediate feedback
    logging.basicConfig(level=logging.INFO)
    run_benchmark()
