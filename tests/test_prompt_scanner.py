import unittest
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.prompting.scanner import PromptScanner

class TestPromptScanner(unittest.TestCase):
    def test_scan_returns_list(self):
        prompts = PromptScanner.scan()
        self.assertIsInstance(prompts, list)

    def test_scan_returns_valid_objects(self):
        prompts = PromptScanner.scan()
        if prompts:
            first = prompts[0]
            self.assertIn("id", first)
            self.assertIn("score", first)
            self.assertIn("name", first)
            self.assertIsInstance(first["score"], int)

    def test_context_scoring(self):
        # Without context
        prompts_base = PromptScanner.scan()
        base_scores = {p['id']: p['score'] for p in prompts_base}

        # With context "risk"
        prompts_boosted = PromptScanner.scan(context=["risk"])
        boosted_scores = {p['id']: p['score'] for p in prompts_boosted}

        # Check if any prompt with "risk" in name/content got boosted
        boosted_count = 0
        for pid, score in boosted_scores.items():
            if score > base_scores.get(pid, 0):
                boosted_count += 1

        # We expect at least some prompts to be boosted given the library has risk prompts
        self.assertGreater(boosted_count, 0)

if __name__ == '__main__':
    unittest.main()
