import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure core is in path
sys.path.append(os.getcwd())

# Mock heavy dependencies
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["tensorflow"] = MagicMock()
sys.modules["spacy"] = MagicMock()
sys.modules["tweepy"] = MagicMock()
sys.modules["transformers.pipeline"] = MagicMock()

# Now import
from core.v23_graph_engine.meta_orchestrator import MetaOrchestrator
from core.schemas.v23_5_schema import HyperDimensionalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_deep_dive_fallback_pipeline():
    print("Initializing MetaOrchestrator...")

    # Mock legacy orchestrator to avoid API key errors
    mock_legacy = MagicMock()

    # Initialize with mock
    orchestrator = MetaOrchestrator(legacy_orchestrator=mock_legacy)

    # Force fallback by mocking deep_dive_app to None in the module
    with patch("core.v23_graph_engine.meta_orchestrator.deep_dive_app", None):

        # Trigger Deep Dive via keyword
        query = "Perform a deep dive analysis on Tesla"

        print(f"Routing Query: {query}")
        result = await orchestrator.route_request(query)

        # Check if we got a dict
        if not isinstance(result, dict):
            print("FAILED: Result is not a dictionary.")
            import json
            try:
                print(json.dumps(result, default=str))
            except:
                print(result)
            sys.exit(1)

        # Verify structure matches the HyperDimensionalKnowledgeGraph schema
        try:
            print("Validating Schema...")
            model = HyperDimensionalKnowledgeGraph(**result)

            target = model.v23_knowledge_graph.meta.target
            conviction = model.v23_knowledge_graph.nodes.strategic_synthesis.final_verdict.conviction_level

            print(f"Target: {target}")
            print(f"Conviction: {conviction}")

            assert target == query
            assert conviction >= 1
            print("\nPipeline Test Passed: Schema Validation Successful.")

        except Exception as e:
            print(f"Schema Validation Failed: {e}")
            import json
            print(json.dumps(result, indent=2, default=str))
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_deep_dive_fallback_pipeline())
