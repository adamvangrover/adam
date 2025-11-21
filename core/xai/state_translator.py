from core.v23_graph_engine.states import RiskAssessmentState

class ExplainableStateTranslator:
    @staticmethod
    def generate_user_update(state: RiskAssessmentState) -> str:
        """
        Converts raw system state into a reassuring, transparent message for the UI.
        """
        status = state.get('human_readable_status', "Processing...")
        iteration = state.get('iteration_count', 0)
        quality = state.get('quality_score', 0.0)
        
        if iteration == 0:
            return f"🔎 Starting analysis on {state['ticker']}. {status}"

        if state.get('needs_correction'):
            return (
                f"🤔 I detected inconsistencies in the analysis (Quality: {quality:.2f}). "
                f"Self-correcting (Attempt {iteration}). Status: {status}"
            )
        
        if quality >= 0.85:
            return f"✅ Analysis complete. High confidence ({quality:.2f}). Generating final report."
            
        return f"⚙️ Processing... Phase: {status} (Iter: {iteration})"
