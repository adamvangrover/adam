import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8124
SERVER_URL = f"http://localhost:{PORT}/showcase/system_knowledge_graph.html"

def start_server():
    """Starts a simple HTTP server in a background thread."""
    # Ensure we serve from repo root
    root_dir = os.getcwd()
    print(f"Serving from {root_dir}")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass # Silence logs

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_system_knowledge_graph():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        try:
            page.goto(SERVER_URL)
        except Exception as e:
            print(f"Failed to navigate: {e}")
            sys.exit(1)

        # Wait for canvas (vis-network)
        try:
            # vis-network creates a canvas inside the container
            page.wait_for_selector("canvas", timeout=10000)
            print("Canvas found (Graph rendered)")

            # Check for legend stats to ensure data loaded
            page.wait_for_selector("#stat-nodes", timeout=10000)
            # Give a little time for text to populate
            time.sleep(1)
            nodes_count = page.inner_text("#stat-nodes")
            print(f"Nodes count displayed: {nodes_count}")

            if nodes_count == "0":
                 print("Warning: Node count is 0, graph might be empty")

        except Exception as e:
            print(f"Timeout waiting for graph: {e}")
            sys.exit(1)

        # Take screenshot
        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/system_knowledge_graph.png"
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
    verify_system_knowledge_graph()

    # We rely on daemon thread to kill server on exit
    sys.exit(0)
