import asyncio
import unittest
from datetime import datetime

from core.agents.hnasp_agent import HNASPAgent
from core.hnasp.logic_engine import LogicEngine
from core.hnasp.personality import BayesACTEngine
from core.hnasp.state_manager import HNASPStateManager
from core.schemas.hnasp import HNASP, ContextStream, LogicLayer, Meta, PersonaState


class TestHNASP(unittest.TestCase):
    def test_logic_engine(self):
        engine = LogicEngine()
        rule = {"and": [{"==": [{"var": "status"}, "active"]}, {">": [{"var": "amount"}, 100]}]}

        # Case 1: True
        data = {"status": "active", "amount": 150}
        self.assertTrue(engine.validate_rule(rule, data))

        # Case 2: False
        data = {"status": "inactive", "amount": 150}
        self.assertFalse(engine.validate_rule(rule, data))

    def test_bayes_act_engine(self):
        engine = BayesACTEngine()
        state = HNASP(
            meta=Meta(agent_id="test", trace_id="1", timestamp=datetime.now(), model_config={"model": "test"}, security_context={"user_id": "u", "clearance": "low"}),
            persona_state=PersonaState(
                identities={
                    "self": {"label": "test", "fundamental_epa": {"E": 1, "P": 1, "A": 1}},
                    "user": {"label": "user", "fundamental_epa": {"E": 0, "P": 0, "A": 0}}
                },
                dynamics={"current_deflection": 0}
            ),
            logic_layer=LogicLayer(state_variables={}, active_rules={}),
            context_stream=ContextStream(window_id=0, turns=[])
        )

        updated_state = engine.update_persona_state(state.persona_state, "You are stupid", role="user")
        # "stupid" maps to attack (-2.0, 1.5, 1.5)
        # Self transient should move towards it (simplification)
        # Deflection should increase
        self.assertGreater(updated_state.dynamics.current_deflection, 0)

    def test_agent_initialization(self):
        config = {"agent_id": "test_agent"}
        agent = HNASPAgent(config)
        self.assertIsInstance(agent.state_manager, HNASPStateManager)

    def test_agent_execution(self):
        async def run_test():
            config = {"agent_id": "exec_agent"}
            agent = HNASPAgent(config)
            response = await agent.execute(user_input="Hello, can I get a loan?")
            return response

        response = asyncio.run(run_test())
        self.assertIsInstance(response, str)
        self.assertIn("cannot approve", response.lower())

if __name__ == "__main__":
    unittest.main()
