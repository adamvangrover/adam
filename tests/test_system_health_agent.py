import unittest
from unittest.mock import MagicMock, patch
import asyncio
import sys
import os
import importlib.util

# Helper to import module without triggering package init
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Path to the agent file
agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/agents/system_health_agent.py'))

# Import the module directly
sha_module = import_module_from_path("system_health_agent_direct", agent_path)
SystemHealthAgent = sha_module.SystemHealthAgent
AgentInput = sha_module.AgentInput
AgentOutput = sha_module.AgentOutput

class TestSystemHealthAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SystemHealthAgent()
        self.input_data = AgentInput(query="check system health", context={})

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('platform.system')
    def test_execute(self, mock_system, mock_disk, mock_memory, mock_cpu):
        # Mock psutil returns
        mock_cpu.return_value = 15.5

        mock_mem_obj = MagicMock()
        mock_mem_obj.total = 16 * (1024**3)
        mock_mem_obj.available = 8 * (1024**3)
        mock_mem_obj.percent = 50.0
        mock_memory.return_value = mock_mem_obj

        mock_disk_obj = MagicMock()
        mock_disk_obj.total = 500 * (1024**3)
        mock_disk_obj.free = 250 * (1024**3)
        mock_disk_obj.percent = 50.0
        mock_disk.return_value = mock_disk_obj

        mock_system.return_value = "Linux"

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.agent.execute(self.input_data))
        loop.close()

        # Assertions
        self.assertIsInstance(result, AgentOutput)
        self.assertIn("System Health Check", result.answer)
        self.assertIn("15.5%", result.answer)
        self.assertEqual(result.metadata['cpu_percent'], 15.5)
        self.assertEqual(result.metadata['memory_total_gb'], 16.0)
        self.assertEqual(result.metadata['os'], "Linux")

if __name__ == '__main__':
    unittest.main()
