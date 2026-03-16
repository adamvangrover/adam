from core.engine.system2_state import System2State
import logging

async def generate_dcf_model(state: System2State) -> dict:
    """
    Graph Node: Generates a DCF model based on historical data.
    If validation_feedback exists, refines the prompt to correct accounting errors.
    """
    ticker = state.get("company_ticker", "UNKNOWN")
    iteration = state.get("iteration_count", 0)
    feedback = state.get("validation_feedback", [])
    
    logging.info(f"[{ticker}] DCF Generator Node Executing... (Iteration: {iteration})")
    
    # MOCK LLM CALL FOR DCF GENERATION
    # In a real implementation, this invokes the LLM using the structured DCFModelOutput Pydantic schema
    
    # Simulate a generated DCF Model. 
    # To demonstrate Reflexion, we intentionally make Terminal Growth > WACC on Iteration 0
    generated_dcf = {
        "company_ticker": ticker,
        "wacc": 0.08,  # 8% WACC
        "terminal_growth_rate": 0.09 if iteration == 0 else 0.03,  # INTENTIONAL ERROR ON P1
        "assumptions": {
            "revenue_growth_rate": 0.15,
            "operating_margin": 0.35,
            "tax_rate": 0.21,
            "capital_expenditure_margin": 0.05,
            "depreciation_margin": 0.04,
            "change_in_nwc_margin": 0.02
        },
        "projected_fcfs": [10.5, 12.1, 14.0, 16.2, 18.8],
        "terminal_value": 376.0,
        "enterprise_value": 435.0,
        "implied_share_price": 145.0
    }
    
    return {
        "generated_dcf": generated_dcf,
        "iteration_count": iteration + 1
    }
