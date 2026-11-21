import unittest
import ast
from adam_v24.intelligence_layer.architects.evolutionary import EvolutionaryArchitect

class TestEvolutionaryArchitect(unittest.TestCase):
    def test_verify_syntax_valid(self):
        arch = EvolutionaryArchitect()
        valid_code = "def foo(): return 1"
        self.assertTrue(arch.verify_syntax(valid_code))

    def test_verify_syntax_invalid(self):
        arch = EvolutionaryArchitect()
        invalid_code = "def foo(): return 1 ("
        self.assertFalse(arch.verify_syntax(invalid_code))

    def test_propose_refactor_no_file(self):
        arch = EvolutionaryArchitect()
        result = arch.propose_refactor("nonexistent.py", "For", "latency")
        self.assertTrue("Error parsing" in result)

if __name__ == '__main__':
    unittest.main()
