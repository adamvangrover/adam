from langgraph.graph import StateGraph, END
from typing import Literal

from core.engine.system2_state import System2State
from core.engine.nodes.dcf_generator_node import generate_dcf_model
from core.engine.nodes.financial_validation_node import validate_financial_model

def route_validation_feedback(state: System2State) -> Literal["regenerate", "finalize"]:
    """
    Conditional Edge Router:
    Decides whether to loop back for another DCF generation or finalize the graph.
    """
    is_valid = state.get("is_valid", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if is_valid:
        # The financial model passed all constraints.
        return "finalize"
    
    if iteration_count >= max_iterations:
        # We hit the max loops. Force exit to prevent infinite hallucination loops.
        import logging
        logging.error(f"[{state.get('company_ticker')}] Max Iterations ({max_iterations}) reached. Model still invalid. Forcing finalize.")
        return "finalize"
        
    # The model was invalid but we still have iterations left. Loop back.
    return "regenerate"

def build_system2_graph():
    """
    Constructs the cyclic "Neuro-Symbolic Graph" for financial modeling.
    Returns the compiled LangGraph application.
    """
    # 1. Initialize State Graph
    workflow = StateGraph(System2State)
    
    # 2. Add Nodes
    workflow.add_node("dcf_generator", generate_dcf_model)
    workflow.add_node("financial_validator", validate_financial_model)
    
    # 3. Define the Flow (Edges)
    # Entry Point -> DCF Generator
    workflow.set_entry_point("dcf_generator")
    
    # DCF Generator always flows into the Validator Node
    workflow.add_edge("dcf_generator", "financial_validator")
    
    # Validator Node loops conditionally based on feedback routing
    workflow.add_conditional_edges(
        "financial_validator",
        route_validation_feedback,
        {
            "regenerate": "dcf_generator",    # Loop Back (Reflexion)
            "finalize": END                   # Exit point
        }
    )
    
    return workflow.compile()

# Global export of the compiled app
system2_app = build_system2_graph()
