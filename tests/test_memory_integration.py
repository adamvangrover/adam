import asyncio
import logging
import os
import json
from unittest.mock import MagicMock
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.data_retrieval_agent import DataRetrievalAgent
from core.system.memory_manager import VectorMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)

async def run_integration_test():
    print("--- Starting Memory Integration Test (Stable) ---")

    test_memory_file = "data/memory/test_history_stable.json"
    if os.path.exists(test_memory_file):
        os.remove(test_memory_file)

    dra_config = {"persona": "Data Retriever"}
    dra = DataRetrievalAgent(dra_config)

    faa_config = {"persona": "Analyst"}
    faa = FundamentalAnalystAgent(faa_config)
    # FundamentalAnalystAgent uses VectorMemoryManager by default now, but we override file
    faa.memory_manager = VectorMemoryManager(storage_file=test_memory_file)

    async def mock_send_message(target_agent, message, timeout=30.0):
        if target_agent == "DataRetrievalAgent":
            return await dra.receive_message("FundamentalAnalystAgent", message)
        return None

    faa.send_message = mock_send_message
    faa.peer_agents['DataRetrievalAgent'] = dra

    # Run with ABC_TEST (Mock Data)
    print("Executing Run 1...")
    result1 = await faa.execute("ABC_TEST")
    assert result1["company_id"] == "ABC_TEST"

    history = faa.memory_manager.query_history("ABC_TEST")
    assert len(history) == 1

    print("Executing Run 2...")
    result2 = await faa.execute("ABC_TEST")
    history2 = faa.memory_manager.query_history("ABC_TEST")
    assert len(history2) == 2

    # Test Vector Search
    print("Testing Vector Search...")
    # Add a dummy entry for another company
    faa.memory_manager.save_analysis("XYZ_TEST", "Analysis of ABC_TEST competitor. Very similar financials. Tech sector.", {})

    # Search similar to ABC_TEST
    # Note: TF-IDF might need more data or distinctive words to match well, but exact words should match.
    similar = faa.memory_manager.search_similar("ABC_TEST analysis", limit=5)
    print(f"Similar found: {len(similar)}")
    if similar:
         print(f"Top match: {similar[0]['company_id']} score: {similar[0].get('similarity_score')}")

    # Should find at least something (ABC_TEST entries themselves match perfectly)
    assert len(similar) > 0

    print("--- Test Passed! ---")
    if os.path.exists(test_memory_file):
        os.remove(test_memory_file)

if __name__ == "__main__":
    asyncio.run(run_integration_test())
