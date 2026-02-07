import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8124
SERVER_URL = f"http://localhost:{PORT}/showcase/system_brain.html"

def start_server():
    """Starts a simple HTTP server in a background thread."""
    root_dir = os.getcwd()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_system_brain():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        page.goto(SERVER_URL)

        # Wait for Chart.js canvases to be present
        try:
            page.wait_for_selector("#memoryChart", timeout=5000)
            print("Memory Chart found")
            page.wait_for_selector("#agent-list li", timeout=5000)
            print("Agent list populated")
        except Exception as e:
            print(f"Timeout waiting for elements: {e}")

        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/system_brain.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    verify_system_brain()
    sys.exit(0)
