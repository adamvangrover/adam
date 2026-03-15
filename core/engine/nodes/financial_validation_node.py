from core.engine.system2_state import System2State
import logging

async def validate_financial_model(state: System2State) -> dict:
    """
    Graph Node: The Reflector.
    Checks the generated DCF model against strict financial accounting rules.
    """
    ticker = state.get("company_ticker", "UNKNOWN")
    dcf = state.get("generated_dcf", {})
    iteration = state.get("iteration_count", 1)
    
    logging.info(f"[{ticker}] Validation Reflector Node Executing...")
    
    feedback = []
    is_valid = True
    
    wacc = dcf.get("wacc", 0)
    tg = dcf.get("terminal_growth_rate", 0)
    
    # --- RULE 1: Terminal Growth MUST be less than WACC ---
    if tg >= wacc:
        error_msg = f"CRITICAL ERROR: Terminal Growth Rate ({tg*100:.1f}%) cannot be greater than or equal to WACC ({wacc*100:.1f}%). This implies the company will outgrow the global economy forever and results in infinite valuation. Recalculate with TG < WACC."
        feedback.append(error_msg)
        is_valid = False
        logging.warning(error_msg)
        
    # --- RULE 2: Operating Margin Bounds ---
    assumptions = dcf.get("assumptions", {})
    op_margin = assumptions.get("operating_margin", 0)
    if op_margin < 0 or op_margin > 1.0:
        error_msg = f"ERROR: Operating Margin ({op_margin*100:.1f}%) is outside logical bounds (0% - 100%)."
        feedback.append(error_msg)
        is_valid = False
        logging.warning(error_msg)
        
    # Compile results
    if is_valid:
        logging.info(f"[{ticker}] DCF Model Passed All Validation Constraints.")
        final_report = f"# System 2 Validated Output: {ticker}\nGenerated Enterprise Value: ${dcf.get('enterprise_value')}B\nValidation passed after {iteration} iterations."
    else:
        logging.warning(f"[{ticker}] DCF Model Failed Validation. Generating Reflexion Feedback.")
        final_report = "Model generation failed validation constraints."

    return {
        "is_valid": is_valid,
        "validation_feedback": feedback,
        "final_report": final_report
    }
