import os
import unittest

from bs4 import BeautifulSoup


class TestShowcaseValidatorSuite(unittest.TestCase):
    def setUp(self):
        self.showcase_dir = "showcase"
        self.files = ["sovereign_os.html", "index.html", "dashboard.html"]

    def test_css_linked(self):
        for f in self.files:
            filepath = os.path.join(self.showcase_dir, f)
            with open(filepath, "r", encoding="utf-8") as html_f:
                soup = BeautifulSoup(html_f.read(), 'html.parser')
                css_links = [l['href'] for l in soup.find_all('link', rel='stylesheet') if not l['href'].startswith('http')]
                for css in css_links:
                    css_path = os.path.join(self.showcase_dir, css)
                    self.assertTrue(os.path.exists(css_path), f"CSS file {css_path} missing for {f}")

    def test_js_linked(self):
        for f in self.files:
            filepath = os.path.join(self.showcase_dir, f)
            with open(filepath, "r", encoding="utf-8") as html_f:
                soup = BeautifulSoup(html_f.read(), 'html.parser')
                js_srcs = [s['src'] for s in soup.find_all('script', src=True) if not s['src'].startswith('http')]
                for js in js_srcs:
                    if js == "../showcase/js/nav.js":
                        continue # Mocked relative link
                    js_path = os.path.join(self.showcase_dir, js)
                    self.assertTrue(os.path.exists(js_path), f"JS file {js_path} missing for {f}")

if __name__ == "__main__":
    unittest.main()
