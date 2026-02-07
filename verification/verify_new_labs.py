import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8128
URLS = [
    f"http://localhost:{PORT}/showcase/scenario_lab.html",
    f"http://localhost:{PORT}/showcase/policy_center.html"
]

def start_server():
    root_dir = os.getcwd()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_labs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for url in URLS:
            print(f"Navigating to {url}")
            page.goto(url)

            try:
                # Scenario Lab check
                if "scenario_lab" in url:
                    page.wait_for_selector("#scatterChart", timeout=5000)
                    print("Chart found in Scenario Lab.")

                # Policy Center check
                if "policy_center" in url:
                    page.wait_for_selector("#snc-container div", timeout=5000)
                    print("Rules found in Policy Center.")

            except Exception as e:
                print(f"Timeout verifying {url}: {e}")

            name = url.split("/")[-1].replace(".html", "")
            os.makedirs("verification", exist_ok=True)
            page.screenshot(path=f"verification/{name}.png")
            print(f"Screenshot saved: verification/{name}.png")

        browser.close()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    verify_labs()
    sys.exit(0)
