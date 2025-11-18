from core.v23_graph_engine.states import RiskAssessmentState

class ExplainableStateTranslator:
    @staticmethod
    def generate_user_update(state: RiskAssessmentState) -> str:
        """
        Converts raw system state into a reassuring, transparent message for the UI.
        """
        if state['iteration_count'] == 0:
            return f"🔎 Starting analysis on {state['ticker']}. Checking primary data sources..."
        
        if state['needs_correction']:
            return (
                f"🤔 I detected an inconsistency in the draft regarding {state['ticker']}'s debt figures. "
                f"Self-correcting (Attempt {state['iteration_count']})..."
            )
        
        if state['quality_score'] > 0.8:
            return f"✅ Analysis complete. High confidence ({state['quality_score']:.2f}). Generatng final report."
            
        return f"⚙️ Processing... Current Phase: {state['human_readable_status']}"

# Example Usage for UI Polling
# update_msg = ExplainableStateTranslator.generate_user_update(current_state)
# print(update_msg)
