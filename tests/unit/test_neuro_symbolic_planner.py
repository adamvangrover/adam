import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner, PlannerIntent

@pytest.fixture
def planner():
    # Mock the UnifiedKnowledgeGraph to prevent actual initialization
    with patch('core.engine.neuro_symbolic_planner.UnifiedKnowledgeGraph') as MockUKG:
        mock_kg = MockUKG.return_value
        mock_kg.graph = nx.DiGraph()
        planner = NeuroSymbolicPlanner()
        # planner.kg is already the mock_kg instance
        return planner

def test_classify_intent_semantic(planner):
    # Test DEEP_DIVE intent
    intent = planner._classify_intent_semantic("Perform a comprehensive deep dive on Apple")
    assert intent == PlannerIntent.DEEP_DIVE

    # Test RISK_ALERT intent
    intent = planner._classify_intent_semantic("Check for risk exposure and credit default")
    assert intent == PlannerIntent.RISK_ALERT

    # Test MARKET_UPDATE intent
    intent = planner._classify_intent_semantic("What is the latest market news and price trend?")
    assert intent == PlannerIntent.MARKET_UPDATE

    # Test GENERAL_QUERY fallback
    intent = planner._classify_intent_semantic("Hello world")
    assert intent == PlannerIntent.GENERAL_QUERY

def test_extract_entities_dynamic(planner):
    # Test regex extraction
    request = "Analyze Tesla Inc. risk profile"
    entities = planner._extract_entities_dynamic(request)
    assert entities.get("primary_entity") == "Tesla Inc."

    # Test heuristic fallback
    request = "Check AAPL status"
    entities = planner._extract_entities_dynamic(request)
    assert entities.get("primary_entity") == "AAPL"

    # Test empty
    request = "Check nothing"
    entities = planner._extract_entities_dynamic(request)
    assert entities == {}

def test_discover_plan_success(planner):
    # Setup mock graph
    planner.kg.graph.add_node("Apple", name="Apple")
    planner.kg.graph.add_node("Risk", name="Risk")
    planner.kg.graph.add_edge("Apple", "Risk", relation="HAS_RISK")

    # Mock get_edge_data
    planner.kg.graph.get_edge_data = MagicMock(return_value={"relation": "HAS_RISK"})

    plan = planner.discover_plan("Apple", "Risk")

    assert plan["symbolic_plan"] is not None
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["description"].startswith("Verify relationship: Apple --[HAS_RISK]--> Risk")

def test_discover_plan_fuzzy_match(planner):
    # Setup mock graph with casing difference
    planner.kg.graph.add_node("Apple Inc.", name="Apple Inc.")
    planner.kg.graph.add_node("Risk", name="Risk")
    planner.kg.graph.add_edge("Apple Inc.", "Risk", relation="HAS_RISK")

    planner.kg.graph.get_edge_data = MagicMock(return_value={"relation": "HAS_RISK"})

    # Request "apple inc." (lowercase)
    plan = planner.discover_plan("apple inc.", "Risk")

    assert plan["steps"][0]["description"].startswith("Verify relationship: Apple Inc. --[HAS_RISK]--> Risk")

def test_discover_plan_fallback(planner):
    # Setup mock graph with no path
    planner.kg.graph.add_node("Apple", name="Apple")
    planner.kg.graph.add_node("Oranges", name="Oranges")
    # No edge

    plan = planner.discover_plan("Apple", "Oranges")

    assert plan["symbolic_plan"] == "RAG_Fallback_Anchored"
    assert len(plan["steps"]) > 0

def test_parse_natural_language_plan(planner):
    text = """
    Here is the plan:
    1. Collect financial data.
    2. Analyze credit risk.
    3. Generate report.
    """
    plan = planner.parse_natural_language_plan(text)

    assert len(plan["steps"]) == 3
    assert plan["steps"][0]["task_id"] == "1"
    assert plan["steps"][0]["description"] == "Collect financial data."

def test_topological_sort(planner):
    steps = [
        {"task_id": "3", "dependencies": ["2"]},
        {"task_id": "1", "dependencies": []},
        {"task_id": "2", "dependencies": ["1"]},
    ]

    sorted_steps = planner._topological_sort(steps)

    ids = [s["task_id"] for s in sorted_steps]
    assert ids == ["1", "2", "3"]

def test_topological_sort_cycle(planner):
    # Cycle: 1->2->1
    steps = [
        {"task_id": "1", "dependencies": ["2"]},
        {"task_id": "2", "dependencies": ["1"]},
    ]

    # Should not infinite loop and should return something
    sorted_steps = planner._topological_sort(steps)
    assert len(sorted_steps) == 2

def test_create_plan_end_to_end(planner):
    # Mock extract_entities to return nothing to force NLP fallback
    with patch.object(planner, '_extract_entities_dynamic', return_value={}):
        request = """
        Analyze AAPL.
        1. Fetch data.
        2. Compute metrics.
        """
        plan = planner.create_plan(request)
        assert plan["symbolic_plan"] == "Natural Language Parsed"
        assert len(plan["steps"]) == 2
