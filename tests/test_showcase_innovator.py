import unittest
from pathlib import Path

from bs4 import BeautifulSoup


class TestShowcaseInnovator(unittest.TestCase):
    def setUp(self):
        self.showcase_dir = Path("showcase")
        self.files_to_test = [
            "sovereign_os.html",
            "index.html",
            "dashboard.html"
        ]

    def test_ai_hooks_exist(self):
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                body_tag = soup.find('body')
                # AI should be able to read system state quickly off the body
                self.assertTrue(body_tag.has_attr('data-ai-state'), f"Body missing data-ai-state in {file}")

                # Ensure meta tags with system context exist
                meta_tags = soup.find_all('meta', attrs={"name": "adam-module-type"})
                self.assertGreaterEqual(len(meta_tags), 1, f"Missing adam-module-type meta tag in {file}")

if __name__ == "__main__":
    unittest.main()
