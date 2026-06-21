import json

class Flywheel:
    def __init__(self):
        self.trace_ledger = "prompt_matrix.jsonl"

    def execute_trace(self, trace_data):
        with open(self.trace_ledger, "a") as f:
            f.write(json.dumps(trace_data) + "\n")

    def evaluate_state_transition(self, prior_state, current_state):
        # Layer 5 (State Transition Evaluation)
        return 1.0 if current_state > prior_state else 0.0

    def evaluate_information_gain(self, trace):
        # Layer 6 (Information Gain Metric)
        new_facts = trace.get("new_facts_extracted", 0)
        tool_calls = trace.get("tool_calls_used", 1)
        tool_calls = max(tool_calls, 1) # Prevent division by zero
        return new_facts / tool_calls
