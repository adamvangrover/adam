import asyncio
from core.engine.system2_state import System2State
from core.engine.system2_graph import system2_app
import pytest

@pytest.mark.asyncio
async def test_system2_reflexion_loop():
    initial_state = {
        "company_ticker": "TEST_TICKER",
        "historical_data": {},
        "iteration_count": 0,
        "max_iterations": 3,
        "validation_feedback": [],
        "is_valid": False
    }
    final_state = await system2_app.ainvoke(initial_state)
    assert final_state["company_ticker"] == "TEST_TICKER"
    assert final_state["iteration_count"] > 1
    assert final_state["is_valid"] is True
    assert "Generated Enterprise Value" in final_state["final_report"]

@pytest.mark.asyncio
async def test_system2_max_iterations_forced_exit():
    from langgraph.graph import StateGraph, END
    from core.engine.nodes.dcf_generator_node import generate_dcf_model
    from core.engine.nodes.financial_validation_node import validate_financial_model
    from core.engine.system2_graph import route_validation_feedback

    async def bad_generator(state):
        return {
            "generated_dcf": {
                "company_ticker": state.get("company_ticker", ""),
                "wacc": 0.08,
                "terminal_growth_rate": 0.10,
                "assumptions": { "operating_margin": 0.35 }
            },
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    workflow = StateGraph(System2State)
    workflow.add_node("dcf_generator", bad_generator)
    workflow.add_node("financial_validator", validate_financial_model)
    workflow.set_entry_point("dcf_generator")
    workflow.add_edge("dcf_generator", "financial_validator")
    workflow.add_conditional_edges(
        "financial_validator",
        route_validation_feedback,
        {"regenerate": "dcf_generator", "finalize": END}
    )
    test_app = workflow.compile()

    initial_state = {
        "company_ticker": "BAD_TICKER",
        "iteration_count": 0,
        "max_iterations": 2,
        "is_valid": False
    }

    final_state = await test_app.ainvoke(initial_state)
    assert final_state["is_valid"] is False
    assert final_state["iteration_count"] >= 2
    assert "failed validation constraints" in final_state["final_report"]

if __name__ == "__main__":
    asyncio.run(test_system2_reflexion_loop())
    asyncio.run(test_system2_max_iterations_forced_exit())
    print("Tests passed.")
