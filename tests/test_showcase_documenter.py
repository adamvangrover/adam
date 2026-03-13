import re
import unittest
from pathlib import Path


class TestShowcaseDocumenter(unittest.TestCase):
    def setUp(self):
        self.showcase_dir = Path("showcase")
        self.files_to_test = [
            "sovereign_os.html",
            "index.html",
            "dashboard.html"
        ]
        self.js_files = [
            "js/sovereign-os.js"
        ]

    def test_inline_comments(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Ensure we have HTML comments
                self.assertIsNotNone(re.search(r'<!--.*?-->', content, re.DOTALL), f"No HTML comments found in {file}")

    def test_js_comments(self):
        for file in self.js_files:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Ensure multi-line doc comment exists
                self.assertIsNotNone(re.search(r'/\*\*?.*?===.*?\*/', content, re.DOTALL), f"No JS header docstring in {file}")

if __name__ == "__main__":
    unittest.main()
