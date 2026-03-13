import unittest
from pathlib import Path

from bs4 import BeautifulSoup


class TestShowcaseModern(unittest.TestCase):
    def setUp(self):
        self.showcase_dir = Path("showcase")
        self.files_to_test = [
            "sovereign_os.html",
            "index.html",
            "dashboard.html"
        ]

    def test_semantic_html5(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                if file != "dashboard.html":
                    self.assertIsNotNone(soup.find('main'), f"Missing <main> tag in {file}")
                    self.assertIsNotNone(soup.find('nav'), f"Missing <nav> tag in {file}")

    def test_aria_labels(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                buttons = soup.find_all('button')
                for btn in buttons:
                    has_text = bool(btn.get_text(strip=True))
                    has_aria = bool(btn.get('aria-label') or btn.get('aria-labelledby'))
                    self.assertTrue(has_text or has_aria, f"Button without accessible name in {file}")

if __name__ == "__main__":
    unittest.main()
