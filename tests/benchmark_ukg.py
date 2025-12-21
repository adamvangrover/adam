import time
import sys
import os
import networkx as nx

# Add repo root to path
sys.path.append(os.getcwd())

from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph

def benchmark():
    ukg = UnifiedKnowledgeGraph()

    # Create a heavy payload
    n_covenants = 100
    risk_state = {
        "ticker": "AAPL",
        "balance_sheet": {"fiscal_year": "2023", "total_debt": 1000},
        "income_statement": {"consolidated_ebitda": 500},
        "covenants": [
            {"name": f"Cov_{i}", "threshold": 4.5, "operator": "<="}
            for i in range(n_covenants)
        ],
        "draft_memo": {"recommendation": "Buy", "confidence_score": 0.9}
    }

    print(f"Verifying correctness with {n_covenants} covenants...")
    ukg.ingest_risk_state(risk_state)

    # Verify nodes
    facility_id = f"CreditFacility::AAPL::General"
    if not ukg.graph.has_node(facility_id):
        print("FAIL: CreditFacility node missing!")
        sys.exit(1)

    for i in range(n_covenants):
        cov_id = f"Covenant::AAPL::Cov_{i}"
        if not ukg.graph.has_node(cov_id):
             print(f"FAIL: Covenant {cov_id} missing!")
             sys.exit(1)
        if not ukg.graph.has_edge(facility_id, cov_id):
             print(f"FAIL: Edge from Facility to {cov_id} missing!")
             sys.exit(1)

    print("CORRECTNESS CHECK PASSED.")

    # Now Benchmark
    ukg_bench = UnifiedKnowledgeGraph()
    n_covenants_bench = 100000
    risk_state_bench = {
        "ticker": "AAPL",
        "balance_sheet": {"fiscal_year": "2023", "total_debt": 1000},
        "income_statement": {"consolidated_ebitda": 500},
        "covenants": [
            {"name": f"Cov_{i}", "threshold": 4.5, "operator": "<="}
            for i in range(n_covenants_bench)
        ],
        "draft_memo": {"recommendation": "Buy", "confidence_score": 0.9}
    }

    print(f"Benchmarking with {n_covenants_bench} covenants...")
    start_time = time.time()
    ukg_bench.ingest_risk_state(risk_state_bench)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
