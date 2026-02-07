import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8127
SERVER_URL = f"http://localhost:{PORT}/showcase/wargame_dashboard.html"

def start_server():
    root_dir = os.getcwd()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_wargame_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        page.goto(SERVER_URL)

        try:
            # Wait for JS to fetch and render
            page.wait_for_selector("#red-log div", timeout=10000)
            print("Log entries rendered.")

            # Check text
            status = page.locator("#battle-status").inner_text()
            print(f"Battle Status: {status}")

        except Exception as e:
            print(f"Timeout waiting for wargame UI: {e}")

        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/wargame.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    verify_wargame_ui()
    sys.exit(0)
