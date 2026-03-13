import unittest
from pathlib import Path

from bs4 import BeautifulSoup


class TestShowcaseHTML(unittest.TestCase):
    def setUp(self):
        self.showcase_dir = Path("showcase")
        self.files_to_test = [
            "sovereign_os.html",
            "index.html",
            "dashboard.html"
        ]

    def test_files_exist(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            self.assertTrue(filepath.exists(), f"File does not exist: {filepath}")

    def test_html_syntax(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                # Check for basic html structure
                self.assertIsNotNone(soup.html, f"Missing <html> tag in {file}")
                self.assertIsNotNone(soup.head, f"Missing <head> tag in {file}")
                self.assertIsNotNone(soup.body, f"Missing <body> tag in {file}")

    def test_no_duplicate_ids(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                ids = [tag.get('id') for tag in soup.find_all(id=True)]
                duplicates = set([x for x in ids if ids.count(x) > 1])
                self.assertEqual(len(duplicates), 0, f"Duplicate IDs found in {file}: {duplicates}")

if __name__ == "__main__":
    unittest.main()
