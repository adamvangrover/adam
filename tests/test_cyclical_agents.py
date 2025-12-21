# tests/test_cyclical_agents.py
import unittest
import asyncio
from unittest.mock import Mock, patch, call

from core.agents.cyclical_reasoning_agent import CyclicalReasoningAgent
from core.agents.reflector_agent import ReflectorAgent


class TestCyclicalAgents(unittest.TestCase):
    def test_reflector_agent(self):
        async def run_test():
            # Mock the MessageBroker
            with patch('core.system.v22_async.async_agent_base.MessageBroker') as MockMessageBroker:
                mock_broker_instance = MockMessageBroker.get_instance.return_value

                # Instantiate the agent
                agent = ReflectorAgent(config={})

                # Define a task
                task = {'payload': 'test'}

                # Execute the agent
                result = await agent.execute(task)

                # Assert the result
                self.assertIn("quality_score", result)
                self.assertIn("critique_notes", result)

        # Run the async test
        asyncio.run(run_test())

    def test_cyclical_reasoning_agent_single_iteration(self):
        async def run_test():
            # Mock the MessageBroker
            with patch('core.system.v22_async.async_agent_base.MessageBroker') as MockMessageBroker:
                mock_broker_instance = MockMessageBroker.get_instance.return_value

                # Instantiate the agent
                agent = CyclicalReasoningAgent(config={})

                # Define a task for a single iteration
                task = {
                    'iterations_left': 1,
                    'payload': {'data': 'initial'},
                    'target_agent': 'ReflectorAgent',
                    'final_reply_to': 'original_caller'
                }

                # Execute the agent
                await agent.execute(task)

                # Assert that send_message was called on the broker with the correct arguments
                mock_broker_instance.publish.assert_called_once()
                args, _ = mock_broker_instance.publish.call_args
                self.assertEqual(args[0], 'ReflectorAgent')

        # Run the async test
        asyncio.run(run_test())

    def test_cyclical_reasoning_agent_termination(self):
        async def run_test():
            # Mock the MessageBroker
            with patch('core.system.v22_async.async_agent_base.MessageBroker') as MockMessageBroker:
                mock_broker_instance = MockMessageBroker.get_instance.return_value

                # Instantiate the agent
                agent = CyclicalReasoningAgent(config={})

                # Define a task for termination
                task = {
                    'iterations_left': 0,
                    'payload': {'data': 'final'},
                    'target_agent': 'ReflectorAgent',
                    'final_reply_to': 'original_caller'
                }

                # Execute the agent
                await agent.execute(task)

                # Assert that send_message was called on the broker with the correct arguments
                mock_broker_instance.publish.assert_called_once()
                args, _ = mock_broker_instance.publish.call_args
                self.assertEqual(args[0], 'original_caller')

        # Run the async test
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
