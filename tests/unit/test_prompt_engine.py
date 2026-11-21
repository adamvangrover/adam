import unittest
import os
import shutil
from pathlib import Path
from backend.intelligence.prompt_engine import PromptEngine

class TestPromptEngine(unittest.TestCase):
    def setUp(self):
        # Create a temporary prompt library structure for testing
        self.test_lib_root = "test_prompt_library"
        os.makedirs(os.path.join(self.test_lib_root, "system"), exist_ok=True)
        os.makedirs(os.path.join(self.test_lib_root, "tasks"), exist_ok=True)

        # Create Core Constitution
        self.core_content = """# Core System
Role: You are a system.
Inputs: {input}
Context: {context}
"""
        with open(os.path.join(self.test_lib_root, "system", "agent_core.md"), "w") as f:
            f.write(self.core_content)

        # Create Derived Prompt
        self.task_content = """
---
# INHERITS: prompt_library/system/agent_core.md
# TASK_TYPE: Test Task

## MISSION
Do the test.
"""
        with open(os.path.join(self.test_lib_root, "tasks", "test_task.md"), "w") as f:
            f.write(self.task_content)

        # Create Derived Prompt with Missing Parent
        self.broken_task_content = """
---
# INHERITS: prompt_library/system/missing_core.md
# TASK_TYPE: Broken Task
"""
        with open(os.path.join(self.test_lib_root, "tasks", "broken_task.md"), "w") as f:
            f.write(self.broken_task_content)

        self.engine = PromptEngine(prompt_library_root=self.test_lib_root)

    def tearDown(self):
        if os.path.exists(self.test_lib_root):
            shutil.rmtree(self.test_lib_root)

    def test_load_simple_prompt(self):
        content = self.engine.load_prompt("system/agent_core.md", {"input": "test input", "context": "test context"})
        self.assertIn("Role: You are a system.", content)

    def test_inheritance_and_chaining(self):
        content = self.engine.load_prompt("tasks/test_task.md", {"input": "test input", "context": "test context"})
        self.assertIn("# Core System", content)
        self.assertIn("## MISSION", content)

    def test_variable_injection(self):
        context = {
            "input": "User query",
            "context": "System state"
        }
        content = self.engine.load_prompt("system/agent_core.md", context)
        self.assertIn("Inputs: User query", content)

    def test_missing_variables_cleaned(self):
        content_with_placeholders = "Memory: {memory}\nTools: {tools}"
        path = os.path.join(self.test_lib_root, "test_placeholders.md")
        with open(path, "w") as f:
            f.write(content_with_placeholders)

        rendered = self.engine.load_prompt("test_placeholders.md", {})
        self.assertNotIn("{memory}", rendered)
        self.assertNotIn("{tools}", rendered)
        self.assertIn("Memory: ", rendered)

    def test_missing_parent_raises_error(self):
        # Should raise FileNotFoundError because 'system/missing_core.md' does not exist
        with self.assertRaises(FileNotFoundError):
            self.engine.load_prompt("tasks/broken_task.md", {})

if __name__ == '__main__':
    unittest.main()
