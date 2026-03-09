#!/usr/bin/env python3
"""
# Architecture & Usage

This script is the performance benchmarking harness for the Adam OS Unified Knowledge Graph (UKG).
It ensures the system remains robust during high-volume node ingestion and query pathfinding.

Usage:
    `uv run python scripts/benchmark_adam.py`

Features:
    - Benchmarks the ingestion of a synthetic risk state containing covenants.
    - Benchmarks knowledge graph pathfinding algorithms between an entity and its deepest covenant node.
    - Outputs system analytics locally via SwarmLogger and the Dummy LLM engine for insight generation.
"""
import time
import asyncio
import sys
import subprocess
import logging
from pathlib import Path

# Add repo root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.llm.engines.dummy_llm_engine import DummyLLMEngine
from core.utils.logging_utils import SwarmLogger

def get_git_revision_hash() -> str:
    """
    Retrieves the current git revision hash for logging.

    Returns:
        str: The git short hash or "unknown" if the git repository is not accessible.
    """
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

class BenchmarkRunner:
    """
    A unified runner for benchmarking Knowledge Graph operational capabilities.

    Attributes:
        logger (SwarmLogger): The telemetry event logger instance.
        git_hash (str): The current repository commit hash.
        ukg_bench (UnifiedKnowledgeGraph): The knowledge graph instance being benchmarked.
        n_covenants (int): The volume of node components to synthetically load into memory.
    """

    def __init__(self, n_covenants: int = 5000) -> None:
        """Initializes the benchmark harness with default constraints."""
        self.logger = SwarmLogger()
        self.git_hash: str = get_git_revision_hash()
        self.ukg_bench = UnifiedKnowledgeGraph()
        self.n_covenants: int = n_covenants

    def setup(self) -> None:
        """Sets up telemetry logs before testing."""
        print(f"Starting Adam Benchmark (Commit: {self.git_hash})...")
        self.logger.log_event("BENCHMARK_START", "BenchmarkRunner", {"git_hash": self.git_hash})

    def run_ingestion_benchmark(self) -> bool:
        """
        Ingests the synthetic node cluster into the knowledge graph structure.

        Returns:
            bool: True if ingestion completed without system exceptions, False otherwise.
        """
        try:
            risk_state_bench = {
                "ticker": "BENCH_TEST",
                "balance_sheet": {"fiscal_year": "2023", "total_debt": 1000},
                "income_statement": {"consolidated_ebitda": 500},
                "covenants": [
                    {"name": f"Cov_{i}", "threshold": 4.5, "operator": "<="}
                    for i in range(self.n_covenants)
                ],
                "draft_memo": {"recommendation": "Buy", "confidence_score": 0.9}
            }

            print(f"Benchmarking UKG Ingestion ({self.n_covenants} covenants)...")
            start_time = time.time()
            self.ukg_bench.ingest_risk_state(risk_state_bench)
            end_time = time.time()

            duration = end_time - start_time
            print(f"Ingestion Time: {duration:.4f} seconds")

            self.logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
                "git_hash": self.git_hash,
                "benchmark_name": "UKG_Ingestion",
                "iterations": self.n_covenants,
                "duration_seconds": duration,
                "status": "SUCCESS"
            })
            return True

        except Exception as e:
            logging.error(f"Ingestion Benchmark Failed: {e}")
            self.logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
                "git_hash": self.git_hash,
                "benchmark_name": "UKG_Ingestion",
                "status": "FAILED",
                "error": str(e)
            })
            return False

    def run_query_benchmark(self) -> None:
        """
        Issues graph traversal queries to test topological iteration speeds.
        Logs failure directly to telemetry on failure without breaking flow.
        """
        try:
            start_node = "LegalEntity::BENCH_TEST"
            target_node = f"Covenant::BENCH_TEST::Cov_{self.n_covenants-1}"

            print(f"Benchmarking UKG Query (Path from {start_node} to {target_node})...")

            start_time = time.time()
            path = self.ukg_bench.find_symbolic_path(start_node, target_node)
            end_time = time.time()

            duration = end_time - start_time
            print(f"Query Time: {duration:.4f} seconds")

            if not path:
                raise Exception("Path not found (Query returned None)")

            self.logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
                "git_hash": self.git_hash,
                "benchmark_name": "UKG_PathQuery",
                "iterations": 1,
                "duration_seconds": duration,
                "status": "SUCCESS"
            })

        except Exception as e:
            logging.error(f"Query Benchmark Failed: {e}")
            self.logger.log_event("BENCHMARK_RESULT", "BenchmarkRunner", {
                "git_hash": self.git_hash,
                "benchmark_name": "UKG_PathQuery",
                "status": "FAILED",
                "error": str(e)
            })

async def run_benchmark() -> None:
    """Entry point for executing all sub-benchmarks and extracting intelligence analysis."""
    runner = BenchmarkRunner()
    runner.setup()
    if runner.run_ingestion_benchmark():
        runner.run_query_benchmark()

        # Innovator phase: Analyze results using LLM
        print("\n--- AI Benchmark Analysis ---")
        llm_engine = DummyLLMEngine(model_name="benchmark-analyzer-v1")
        prompt = "Analyze these benchmark results for Adam OS Unified Knowledge Graph."
        context = "Ingestion and Query benchmarks ran successfully under moderate load."

        insight = await llm_engine.generate_response(prompt=prompt, context=context)
        print(f"Insight: {insight}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_benchmark())
