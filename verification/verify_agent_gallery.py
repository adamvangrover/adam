import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8126
SERVER_URL = f"http://localhost:{PORT}/showcase/agent_gallery.html"

def start_server():
    root_dir = os.getcwd()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_gallery():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        page.goto(SERVER_URL)

        try:
            page.wait_for_selector("#gallery-container div", timeout=5000)
            count = page.locator("#gallery-container > div").count()
            print(f"Gallery Cards Found: {count}")

            # Check for a specific agent
            if count > 0:
                text = page.locator("#gallery-container").inner_text()
                if "RepoGuardianAgent" in text:
                    print("RepoGuardianAgent found in gallery.")

        except Exception as e:
            print(f"Timeout waiting for gallery: {e}")

        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/agent_gallery.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    verify_gallery()
    sys.exit(0)
