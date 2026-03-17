from typing import TypedDict, Annotated, Sequence, Any, Dict
import operator
from langgraph.graph import StateGraph, END
import logging

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

    return workflow
