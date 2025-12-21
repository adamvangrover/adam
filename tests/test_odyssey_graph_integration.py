from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_odyssey_graph_integration():
    ukg = UnifiedKnowledgeGraph()

    # Mock Risk State (mimicking VerticalRiskGraphState)
    mock_state = {
        "ticker": "ODYSSEY_TEST",
        "balance_sheet": {
            "cash_equivalents": 100.0,
            "total_assets": 1000.0,
            "total_debt": 500.0,
            "equity": 500.0,
            "fiscal_year": 2024
        },
        "income_statement": {
            "revenue": 2000.0,
            "operating_income": 300.0,
            "net_income": 200.0,
            "depreciation_amortization": 50.0,
            "consolidated_ebitda": 350.0
        },
        "covenants": [
            {
                "name": "Max Leverage",
                "threshold": 4.0,
                "operator": "<=",
                "definition_text": "Net Debt / EBITDA",
                "add_backs": []
            }
        ],
        "draft_memo": {
            "recommendation": "BUY",
            "confidence_score": 0.95
        }
    }

    # Run Ingestion
    print("Ingesting Risk State...")
    ukg.ingest_risk_state(mock_state)

    # Verify Nodes
    print(f"Total Nodes: {ukg.graph.number_of_nodes()}")

    nodes = list(ukg.graph.nodes(data=True))
    node_types = [d['type'] for n, d in nodes]

    print(f"Node Types found: {set(node_types)}")

    assert "LegalEntity" in node_types
    assert "FinancialReport" in node_types
    assert "Covenant" in node_types
    assert "CreditFacility" in node_types
    assert "RiskModel" in node_types

    # Verify Edges
    print(f"Total Edges: {ukg.graph.number_of_edges()}")

    # Verify Financial Data
    report_node = [n for n, d in nodes if d['type'] == 'FinancialReport'][0]
    report_data = ukg.graph.nodes[report_node]
    print(f"Report Data: {report_data}")

    # Check Leverage Calculation
    # Debt 500 / EBITDA 350 = 1.428...
    expected_lev = 500.0 / 350.0
    assert abs(report_data['leverage_ratio'] - expected_lev) < 0.001

    print("Verification Successful!")


if __name__ == "__main__":
    test_odyssey_graph_integration()
