import unittest
import os
import shutil
import json
from core.agents.mixins.memory_mixin import MemoryMixin
from datetime import datetime

class TestMemoryMixin(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/memory"
        os.makedirs(self.test_dir, exist_ok=True)
        self.file_path = f"{self.test_dir}/test_agent_memory.json"

        class MockAgent(MemoryMixin):
            name = "test_agent"
            context = {"initial": "value"}

        self.agent = MockAgent()

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_save_and_load_memory(self):
        # Save
        extra_data = {"last_thought": "I think therefore I am"}
        self.agent.save_memory(self.file_path, extra_data)

        # Verify file exists
        self.assertTrue(os.path.exists(self.file_path))

        # Load
        loaded_data = self.agent.load_memory(self.file_path)

        # Check integrity
        self.assertEqual(loaded_data["agent_id"], "test_agent")
        self.assertEqual(loaded_data["last_thought"], "I think therefore I am")
        self.assertEqual(loaded_data["context"]["initial"], "value")

if __name__ == '__main__':
    unittest.main()
