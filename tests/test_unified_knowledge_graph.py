import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

@pytest.fixture
def clean_ukg():
    # Reset singleton if possible or mock the cache
    with patch('core.engine.unified_knowledge_graph.GraphCache') as MockCache:
        mock_cache_instance = MockCache.get_instance.return_value
        mock_cache_instance.get_graph.return_value = None

        # We need to ensure we don't accidentally use a shared instance from other tests
        # The class uses `global _SHARED_GRAPH_INSTANCE`.
        # Since we can't easily reset the global variable in the module from here without importing it
        # we will patch the module global if possible or just rely on the fact that we are mocking GraphCache.

        # Actually, let's just test a fresh instance by mocking the cache to return None initially.
        ukg = UnifiedKnowledgeGraph()
        return ukg

def test_initialization(clean_ukg):
    assert clean_ukg.graph is not None
    assert isinstance(clean_ukg.graph, nx.DiGraph)
    # Check if some default nodes are added (from ingestion)
    # _ingest_fibo_ontology adds nodes
    assert clean_ukg.graph.number_of_nodes() > 0

def test_singleton_behavior():
    with patch('core.engine.unified_knowledge_graph.GraphCache') as MockCache:
        mock_cache_instance = MockCache.get_instance.return_value
        # First call returns None, so it initializes
        mock_cache_instance.get_graph.side_effect = [None, None, nx.DiGraph()]

        ukg1 = UnifiedKnowledgeGraph()

        # Second call returns the graph set by the first call (we mock this behavior)
        mock_cache_instance.get_graph.side_effect = None
        mock_cache_instance.get_graph.return_value = ukg1.graph

        ukg2 = UnifiedKnowledgeGraph()

        assert ukg1.graph is ukg2.graph

def test_find_symbolic_path_success(clean_ukg):
    # Setup a simple path
    clean_ukg.graph.add_node("A", prov_source="Test")
    clean_ukg.graph.add_node("B", prov_source="Test")
    clean_ukg.graph.add_node("C", prov_source="Test")
    clean_ukg.graph.add_edge("A", "B", relation="to")
    clean_ukg.graph.add_edge("B", "C", relation="to")

    path = clean_ukg.find_symbolic_path("A", "C")

    assert path is not None
    assert len(path) == 2
    assert path[0]["source"] == "A"
    assert path[0]["target"] == "B"
    assert path[1]["source"] == "B"
    assert path[1]["target"] == "C"

def test_find_symbolic_path_no_path(clean_ukg):
    clean_ukg.graph.add_node("X")
    clean_ukg.graph.add_node("Y")

    path = clean_ukg.find_symbolic_path("X", "Y")
    assert path is None

def test_query_node_metadata(clean_ukg):
    clean_ukg.graph.add_node("NodeM", type="TestType", attr="Value")

    meta = clean_ukg.query_node_metadata("NodeM")
    assert meta["type"] == "TestType"
    assert meta["attr"] == "Value"

def test_ingest_regulatory_updates(clean_ukg):
    regulations = [
        {"source": "SEC", "title": "New Rule", "summary": "Details"}
    ]
    clean_ukg.ingest_regulatory_updates(regulations)

    node_id = "Regulation::SEC::New Rule"
    assert clean_ukg.graph.has_node(node_id)
    assert clean_ukg.graph.nodes[node_id]["type"] == "Regulation"
