import logging
import operator
from typing import Annotated, Any, Dict, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from core.engine.financial_reflector import financial_validation_reflector, should_recalculate
from scripts.quantum_market_simulator import quantum_simulator_node

logger = logging.getLogger(__name__)

# State schema for our Financial Reflexion Loop
class FinancialState(TypedDict):
    """
    State representing the cyclical financial reasoning and simulation process.
    """
    # The original query or ticker
    target_entity: str

    # Financial context (e.g., current DCF assumptions, historical data)
    financial_context: Dict[str, Any]

    # The output of the simulator (e.g., Monte Carlo paths, forecasted EV)
    simulation_results: Dict[str, Any]

    # Validation status from the reflector
    is_valid: bool

    # Critique notes detailing any accounting violations
    critique_notes: Annotated[Sequence[str], operator.add]

    # Number of reflection iterations to prevent infinite loops
    iteration_count: int

def get_financial_state_machine():
    """
    Initializes and compiles the LangGraph state machine for System 2
    Cyclic Financial Reasoning (Reflexion Loops).
    """
    # Initialize the graph
    workflow = StateGraph(FinancialState)

    # Add Nodes
    workflow.add_node("quantum_simulation", quantum_simulator_node)
    workflow.add_node("financial_validation", financial_validation_reflector)

    # Set entry point
    workflow.set_entry_point("quantum_simulation")

    # Add edges
    workflow.add_edge("quantum_simulation", "financial_validation")

    # Conditional edge for Reflexion Loop
    workflow.add_conditional_edges(
        "financial_validation",
        should_recalculate,
        {
            "recalculate": "quantum_simulation",
            "finalize": END
        }
    )

    return workflow.compile()
