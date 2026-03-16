import asyncio
import logging
import json
import os
import sys

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.memory_orchestrator import MemoryOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_memory_tests():
    """
    Test harness to initialize the Active Memory Engine, load legacy data,
    and execute relational/semantic queries.
    """
    print("\n" + "="*80)
    print("INITIALIZING PRIORITY 2: ACTIVE MEMORY ENGINE (VECTOR + GRAPH)")
    print("="*80 + "\n")
    
    # Initialize the Orchestrator (Will create ChromaDB and NetworkX instances)
    memory = MemoryOrchestrator()
    
    # 1. PREPOPULATE VECTOR DATABASE
    print("\n--- INGESTING HISTORICAL REPORTS INTO VECTOR MEMORY ---")
    mock_files = [
        "showcase/market_mayhem_historical_report.md",
        "showcase/market_mayhem_cro_ib_week_ahead.md"
    ]
    
    for file_path in mock_files:
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path)
        if os.path.exists(full_path):
            memory.prepopulate_memory(full_path, doc_id=os.path.basename(file_path))
        else:
            logging.warning(f"Test File Not Found: {full_path}")
            
    # 2. SEED GRAPH DATABASE RELATIONSHIPS
    print("\n--- SEEDING SYSTEMIC RISK RELATIONSHIPS INTO GRAPH MEMORY ---")
    # Add a custom relationship just for the test
    memory.graph_db.add_relationship("NVDA", "critical_supplier_to", "Hyperscalers", {"weight": 0.9})
    memory.graph_db.add_relationship("Hyperscalers", "highly_exposed_to", "AI Capex Cycle", {"weight": 1.0})
    memory.graph_db.save_graph()
    logging.info("Graph seeded and explicitly saved.")

    # 3. EXECUTE UNIFIED OMNI-QUERY
    print("\n" + "="*80)
    print("EXECUTING OMNI-QUERY: NVDA Supply Chain & AI Capex Risk")
    print("="*80 + "\n")
    
    target_entity = "NVDA"
    thematic_query = "What are the systemic risks involving hyperscalers, AI capex, and supply chains?"
    
    context = await memory.get_total_context(entity=target_entity, thematic_query=thematic_query)
    
    # 4. PRINT RESULTS
    print(f"\n[TARGET]: {context['target_entity']}")
    print(f"[QUERY]: {context['query']}")
    
    print("\n--- GRAPH RELATIONSHIPS (RADIUS 2) ---")
    if not context['structural_relationships']:
        print("  => No structural relationships found.")
    for rel in context['structural_relationships']:
        print(f"  => {rel['synthesized_narrative']}")
        
    print("\n--- VECTOR SEMANTIC RAG HISTORY (TOP 3 CHUNKS) ---")
    if not context['semantic_history']:
        print("  => No semantic history found.")
    for idx, segment in enumerate(context['semantic_history']):
        print(f"\n[Snippet {idx+1}]")
        print(f"\"{segment[:300]}...\"")
        
    print("\nTEST HARNESS COMPLETE.")

if __name__ == "__main__":
    asyncio.run(run_memory_tests())
