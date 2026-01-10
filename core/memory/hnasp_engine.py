import json
from datetime import datetime
from typing import Dict, Any, List
import json_logic
from core.schemas.hnasp import HNASPState, Turn, LogicLayer, Meta, PersonaState, ContextStream, EPAVector, Identity, PersonaIdentities, SecurityContext

class HNASPEngine:
    """
    Middleware engine that intercepts LLM responses, validates logic,
    and manages the HNASP state.
    """
    def __init__(self, state: HNASPState):
        self.state = state

    def update_context(self, role: str, content: str, logic_eval: Dict[str, Any] = None):
        """Adds a new turn to the context stream."""
        turn = Turn(role=role, content=content, logic_eval=logic_eval)
        self.state.context_stream.turns.append(turn)

    def validate_logic(self, data: Dict[str, Any]) -> bool:
        """
        Executes the stored business rules (AST) against the provided data.
        Returns True if logic passes, False otherwise.
        """
        if not self.state.logic_layer.active_rules:
            return True # No rules to enforce

        try:
            result = json_logic.apply(self.state.logic_layer.active_rules, data)

            # Record trace
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "result": result
            }
            self.state.logic_layer.execution_trace.append(trace_entry)

            return bool(result)
        except Exception as e:
            # Log error in trace
            self.state.logic_layer.execution_trace.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            return False

    def persist_state(self, storage_path: str):
        """
        Simulates writing to an Observation Lakehouse (Parquet/JSONL).
        In a real impl, this would use pandas.to_parquet or s3 boto3.
        """
        data = self.state.model_dump(mode='json')
        # Appending to a local JSONL file for now
        with open(storage_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    @staticmethod
    def create_initial_state(agent_id: str, trace_id: str, epa: Dict[str, float]) -> HNASPState:
        # Construct complex nested state using centralized schemas
        return HNASPState(
            meta=Meta(
                agent_id=agent_id,
                trace_id=trace_id,
                security_context=SecurityContext(user_id="system", clearance="public")
            ),
            logic_layer=LogicLayer(),
            persona_state=PersonaState(
                identities=PersonaIdentities(
                    self=Identity(
                        label="Adam",
                        fundamental_epa=EPAVector(**epa),
                        transient_epa=EPAVector(**epa)
                    ),
                    user=Identity(
                        label="User",
                        fundamental_epa=EPAVector(E=0.0, P=0.0, A=0.0)
                    )
                )
            ),
            context_stream=ContextStream()
        )
