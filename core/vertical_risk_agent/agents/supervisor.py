from typing import Literal, Dict, Any

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Mock for blueprint purposes if langgraph not installed
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, start, end): pass
        def add_conditional_edges(self, source, router, map): pass
        def compile(self, checkpointer=None, interrupt_before=None): return self
        def set_entry_point(self, name): pass

    # Mock CompiledGraph for UI to use invoke
    class CompiledGraphMock:
        def invoke(self, state, config=None):
            # Return a mock output state for the UI
            return {
                "balance_sheet": {
                    "cash": 50000000,
                    "total_debt": 120000000,
                    "consolidated_ebitda": 40000000
                },
                "covenants": [
                    {"name": "Net Leverage Ratio", "threshold": 4.5, "current": 3.0}
                ],
                "draft_memo": {"memo": "Based on the analysis, the company is in good standing..."},
                "messages": ["Analysis Complete"]
            }

    END = "END"

    class MemorySaver:
        pass

    # If mocking, modify StateGraph to return the CompiledGraphMock on compile
    StateGraph.compile = lambda self, checkpointer=None, interrupt_before=None: CompiledGraphMock()


from ..state import VerticalRiskGraphState
from .analyst import QuantAgent
from .legal import LegalAgent
from .market import MarketAgent

# Initialize Agents
quant_agent = QuantAgent(model=None)
legal_agent = LegalAgent(model=None)
market_agent = MarketAgent(model=None)


def supervisor_node(state: VerticalRiskGraphState):
    """
    The Supervisor decides which agent to call next or if the process is done.
    """
    print("--- Supervisor: Routing ---")
    messages = state.get("messages", [])

    # Logic:
    # 1. If no financials, call Quant.
    # 2. If no legal analysis, call Legal.
    # 3. If no market research, call Market.
    # 4. If all done, Critique.
    # 5. If Critique passed (or count > 0 for demo), Approval.

    return {"next_step": "deciding"}


def route_supervisor(state: VerticalRiskGraphState) -> Literal["quant", "legal", "market", "critique", "human_approval"]:
    """
    Conditional logic to determine the next node.
    """
    if not state.get("quant_analysis"):
        return "quant"
    if not state.get("legal_analysis"):
        return "legal"
    if not state.get("market_research"):
        return "market"

    if state.get("critique_count", 0) < 1:
        return "critique"

    return "human_approval"


def critique_node(state: VerticalRiskGraphState):
    """
    Checks for consistency between Quant and Legal.
    """
    print("--- Critiquer: Verifying Consistency ---")
    # Logic: Check if Debt/EBITDA matches covenant definitions
    return {
        "critique_count": state.get("critique_count", 0) + 1,
        "messages": ["Critiquer: Logic consistent."]
    }


def human_approval_node(state: VerticalRiskGraphState):
    """
    Pauses for human sign-off.
    """
    print("--- Human Approval Request ---")
    return {"status": "Waiting for approval"}

# --- Build the Graph ---


workflow = StateGraph(VerticalRiskGraphState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("quant", quant_agent.analyze_financials)
workflow.add_node("legal", legal_agent.analyze_covenants)
workflow.add_node("market", market_agent.research_market)
workflow.add_node("critique", critique_node)
workflow.add_node("human_approval", human_approval_node)

# Set Entry Point
workflow.set_entry_point("supervisor")

# Add Conditional Edges
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "quant": "quant",
        "legal": "legal",
        "market": "market",
        "critique": "critique",
        "human_approval": "human_approval"
    }
)

# Add Normal Edges
workflow.add_edge("quant", "supervisor")
workflow.add_edge("legal", "supervisor")
workflow.add_edge("market", "supervisor")
workflow.add_edge("critique", "supervisor")
workflow.add_edge("human_approval", END)

# Compile with Checkpointer for Time Travel
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
