import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from core.schemas.hnasp import HNASPState as HNASP, Meta, PersonaState, LogicLayer, ContextStream, Turn, ModelConfig, SecurityContext, Identity, PersonaIdentities, PersonaDynamics, EPAVector, ExecutionTrace
from core.hnasp.lakehouse import ObservationLakehouse
from core.hnasp.logic_engine import LogicEngine
from core.hnasp.personality import BayesACTEngine
import logging

logger = logging.getLogger(__name__)

class HNASPStateManager:
    def __init__(self,
                 lakehouse: ObservationLakehouse,
                 logic_engine: LogicEngine,
                 personality_engine: BayesACTEngine,
                 llm_client: Any): # generic llm client
        self.lakehouse = lakehouse
        self.logic_engine = logic_engine
        self.personality_engine = personality_engine
        self.llm_client = llm_client

    def initialize_agent(self, agent_id: str, identity_label: str = "assistant", initial_rules: Optional[Dict[str, Any]] = None) -> HNASP:
        """
        Creates a fresh HNASP state for a new agent.
        """
        # Create default EPA
        # (Using values from whitepaper example for 'auditor')
        self_identity = Identity(
            label=identity_label,
            fundamental_epa=EPAVector(E=1.2, P=0.9, A=0.4),
            transient_epa=EPAVector(E=1.2, P=0.9, A=0.4)
        )
        user_identity = Identity(
            label="user",
            fundamental_epa=EPAVector(E=0.0, P=0.0, A=0.0),
            confidence=0.5
        )

        state = HNASP(
            meta=Meta(
                agent_id=agent_id,
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                model_config={
                    "model": "mock-gpt-4",
                    "temperature": 0.7
                },
                security_context=SecurityContext(
                    user_id="system_init",
                    clearance="admin"
                )
            ),
            persona_state=PersonaState(
                identities=PersonaIdentities(self=self_identity, user=user_identity),
                dynamics=PersonaDynamics(current_deflection=0.0)
            ),
            logic_layer=LogicLayer(
                state_variables={},
                active_rules=initial_rules or {}
            ),
            context_stream=ContextStream(
                window_id=0,
                turns=[]
            )
        )
        return state

    def run_cycle(self, agent_id: str, user_input: str, state_variables: Optional[Dict[str, Any]] = None, initial_rules: Optional[Dict[str, Any]] = None) -> str:
        """
        The Load-Think-Save cycle.
        """
        # 1. Load (Rehydration)
        current_state = self.lakehouse.load_latest_state(agent_id)
        if not current_state:
            current_state = self.initialize_agent(agent_id, initial_rules=initial_rules)

        # Update Trace ID and Timestamp
        current_state.meta.trace_id = str(uuid.uuid4())
        current_state.meta.timestamp = datetime.now()

        # 2. Perceive & Update (Personality & Logic)
        # Update Logic Variables (e.g. from DB)
        if state_variables:
            current_state.logic_layer.state_variables.update(state_variables)

        # Update Personality (BayesACT)
        current_state.persona_state = self.personality_engine.update_persona_state(
            current_state.persona_state, user_input, role="user"
        )

        # Record User Turn
        user_turn = Turn(
            role="user",
            timestamp=datetime.now(),
            content=user_input,
            intent="unknown" # NLP intent classification would go here
        )
        current_state.context_stream.turns.append(user_turn)

        # 3. Neurosymbolic Reasoning (The "Think" Step)
        # Serialize state to JSON for the prompt
        system_prompt = current_state.model_dump_json(exclude={'context_stream'})
        # We exclude context stream from the system prompt block to append it as chat history usually,
        # but whitepaper implies the WHOLE thing is the prompt.
        # "The entire HNASP JSON is passed to the LLM as the system prompt."

        full_prompt = current_state.model_dump_json(by_alias=True)

        # Invoke LLM
        # We expect the LLM to return a JSON containing the execution trace and the response.
        # For this prototype, we'll simulate the LLM's logic execution if it's a mock client
        # OR we try to parse it.

        response_payload = self.llm_client.generate(full_prompt, user_input)

        # Parsing response (Assuming LLM returns JSON with 'execution_trace' and 'response_text')
        # If parsing fails, we handle it.
        try:
            # Heuristic: Find JSON block
            if isinstance(response_payload, str):
                # Try to load json
                 llm_output = json.loads(response_payload)
            else:
                 llm_output = response_payload

            generated_trace = llm_output.get("execution_trace")
            response_text = llm_output.get("response_text")

        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")
            generated_trace = None
            response_text = "I'm sorry, I encountered an internal error processing my state."

        # 4. Validation (Logic Guardrails)
        # The backend executes the logic independently
        backend_trace = self.logic_engine.batch_validate(
            current_state.logic_layer.active_rules,
            current_state.logic_layer.state_variables
        )

        # Neurosymbolic Dissonance Check
        if generated_trace:
            claimed_rule_id = generated_trace.get("rule_id")
            claimed_result = generated_trace.get("result")

            # If the LLM claimed to execute a rule that exists
            if claimed_rule_id in backend_trace:
                actual_result = backend_trace[claimed_rule_id]
                # Simple equality check
                if str(actual_result) != str(claimed_result): # Stringify for safe comparison
                    logger.critical(f"Neurosymbolic Dissonance detected! Claimed: {claimed_result}, Actual: {actual_result}")
                    return "SYSTEM ERROR: Logic Verification Failed. Response halted."

        # Logic to "Inject" the backend trace into the state
        current_state.logic_layer.execution_trace = ExecutionTrace(
            rule_id="batch",
            result=backend_trace
        )

        # Record Agent Thought
        thought_turn = Turn(
            role="agent_thought",
            timestamp=datetime.now(),
            logic_eval=backend_trace,
            internal_monologue="Logic validated."
        )
        current_state.context_stream.turns.append(thought_turn)

        # Record Agent Response
        agent_turn = Turn(
            role="assistant",
            timestamp=datetime.now(),
            content=response_text
        )
        current_state.context_stream.turns.append(agent_turn)

        # 5. Save (Persistence)
        self.lakehouse.save_trace(current_state)

        return response_text
