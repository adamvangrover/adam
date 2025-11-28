from typing import Literal
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Mock for blueprint purposes
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, start, end): pass
        def add_conditional_edges(self, source, router, map): pass
        def compile(self, checkpointer=None, interrupt_before=None): return self
    END = "END"
    class MemorySaver: pass

from ..state import VerticalRiskGraphState
from .workers import QuantAgent, LegalAgent, MarketAgent

# Initialize Agents
# In a real app, these would be initialized with an LLM instance (e.g. ChatOpenAI)
quant_agent = QuantAgent(model=None)
legal_agent = LegalAgent(model=None)
market_agent = MarketAgent(model=None)

def supervisor_node(state: VerticalRiskGraphState):
    """
    The Supervisor decides which agent to call next or if the process is done.
    """
    print("--- Supervisor: Routing ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else ""

    # Simple logic for blueprint:
    # If no analysis, start with Quant.
    # If Quant done, do Legal.
    # If Legal done, do Market.
    # If all done, go to Critiquer.

    # In a real implementation, an LLM would make this decision dynamically.
    return {"next_step": "deciding"} # State update

def route_supervisor(state: VerticalRiskGraphState) -> Literal["quant", "legal", "market", "critique", "human_approval"]:
    """
    Conditional logic to determine the next node.
    """
    # Check what data is missing
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
# Human approval is a terminal state or loops back if rejected
workflow.add_edge("human_approval", END)

# Compile with Checkpointer for Time Travel
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
