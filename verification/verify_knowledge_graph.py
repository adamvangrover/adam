import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8123
SERVER_URL = f"http://localhost:{PORT}/showcase/knowledge_graph.html"

def start_server():
    """Starts a simple HTTP server in a background thread."""
    # Ensure we serve from repo root
    # We assume the script is running from repo root or we adjust
    root_dir = os.getcwd()
    print(f"Serving from {root_dir}")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass # Silence logs

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_knowledge_graph():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        page.goto(SERVER_URL)

        # Wait for D3 to render nodes
        # We look for the SVG element or a node class
        try:
            page.wait_for_selector("svg", timeout=10000)
            print("SVG found")
            page.wait_for_selector("circle", timeout=10000)
            print("Nodes found")
        except Exception as e:
            print(f"Timeout waiting for graph: {e}")

        # Take screenshot
        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/knowledge_graph.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give server a moment
    time.sleep(1)

    # Run verification
    verify_knowledge_graph()

    # We rely on daemon thread to kill server on exit
    sys.exit(0)
