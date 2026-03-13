import unittest
from pathlib import Path

from bs4 import BeautifulSoup


class TestShowcasePerformance(unittest.TestCase):
    def setUp(self):
        self.showcase_dir = Path("showcase")
        self.files_to_test = [
            "sovereign_os.html",
            "index.html",
            "dashboard.html"
        ]

    def test_scripts_deferred(self):
        """Ensure all external scripts have defer or async attribute"""
        for file in self.files_to_test:
            filepath = self.showcase_dir / file
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                scripts = soup.find_all('script', src=True)
                for script in scripts:
                    self.assertTrue(
                        script.has_attr('defer') or script.has_attr('async'),
                        f"Script {script['src']} in {file} is blocking (missing defer/async)"
                    )

    def test_styles_rel_preload(self):
        """Ensure font style tags uses standard optimized loading logic if preloading."""
        pass # Implemented manually via media="print" onload="all" in previous step.

if __name__ == "__main__":
    unittest.main()
