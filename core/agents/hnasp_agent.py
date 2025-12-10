from typing import Any, Dict, Optional
import asyncio
import logging
from core.agents.agent_base import AgentBase
from core.hnasp.state_manager import HNASPStateManager
from core.hnasp.lakehouse import ObservationLakehouse
from core.hnasp.logic_engine import LogicEngine
from core.hnasp.personality import BayesACTEngine

logger = logging.getLogger(__name__)

class MockLLMClient:
    """
    A simple mock LLM that pretends to follow HNASP.
    """
    def generate(self, system_prompt: str, user_input: str) -> Dict[str, Any]:
        logger.info(f"MockLLM received prompt length: {len(system_prompt)}")
        logger.info(f"MockLLM received user input: {user_input}")

        # Simple heuristic response
        if "loan" in user_input.lower():
            return {
                "execution_trace": {
                    "rule_id": "loan_approval_policy",
                    "result": False, # Mock rejection
                    "step_by_step": [{"step": "checked amount", "value": "high"}]
                },
                "response_text": "I cannot approve this loan based on current policy."
            }

        return {
            "execution_trace": {
                "rule_id": "default",
                "result": True
            },
            "response_text": f"I processed your request: {user_input}"
        }

class HNASPAgent(AgentBase):
    """
    An agent that implements the Hybrid Neurosymbolic Agent State Protocol (HNASP).
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Any] = None):
        super().__init__(config, constitution, kernel)

        # Initialize HNASP Components
        self.lakehouse = ObservationLakehouse() # Default path
        self.logic_engine = LogicEngine()
        self.personality_engine = BayesACTEngine()
        self.llm_client = MockLLMClient() # In real system, this would be an OpenAI client wrapper

        self.state_manager = HNASPStateManager(
            lakehouse=self.lakehouse,
            logic_engine=self.logic_engine,
            personality_engine=self.personality_engine,
            llm_client=self.llm_client
        )

        self.agent_id = config.get("agent_id", "hnasp_default")
        self.identity_label = config.get("identity_label", "assistant")
        self.active_rules = config.get("active_rules", {})

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the HNASP cycle.
        Expected kwargs:
            user_input: str
            state_variables: Dict[str, Any] (optional)
        """
        user_input = kwargs.get("user_input", "")
        state_variables = kwargs.get("state_variables", {})

        if not user_input:
            logger.warning("HNASPAgent received empty user_input.")
            return "Please provide input."

        try:
            # Run blocking I/O (file access) in a separate thread
            response = await asyncio.to_thread(
                self.state_manager.run_cycle,
                agent_id=self.agent_id,
                user_input=user_input,
                state_variables=state_variables,
                initial_rules=self.active_rules
            )
            return response
        except Exception as e:
            logger.error(f"HNASP Execution failed: {e}")
            raise

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "HNASPAgent",
            "description": "A neurosymbolic agent with deterministic logic and probabilistic personality.",
            "skills": [
                {
                    "name": "execute",
                    "description": "Process user input statefully via HNASP.",
                    "parameters": {
                        "user_input": "The text input from the user.",
                        "state_variables": "Optional dictionary of logic variables (e.g. transaction_amount)."
                    }
                }
            ]
        }
