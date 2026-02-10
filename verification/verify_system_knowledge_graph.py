import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from functools import partial
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8124
REPO_ROOT = os.getcwd()
SERVE_DIR = os.path.join(REPO_ROOT, "services/v24_dashboard/public")
SERVER_URL = f"http://localhost:{PORT}/system_knowledge_graph.html"

def start_server():
    """Starts a simple HTTP server serving SERVE_DIR."""
    print(f"Serving from {SERVE_DIR}")

    # Silence logs
    class SilentHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    # Python 3.7+ supports directory argument in SimpleHTTPRequestHandler
    handler = partial(SilentHandler, directory=SERVE_DIR)

    httpd = HTTPServer(('localhost', PORT), handler)
    httpd.serve_forever()

def verify_system_knowledge_graph():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        try:
            page.goto(SERVER_URL, timeout=10000)
        except Exception as e:
            print(f"Failed to navigate: {e}")
            sys.exit(1)

        # Wait for Vis.js canvas
        try:
            print("Waiting for canvas...")
            page.wait_for_selector("#graph-container canvas", timeout=10000)
            print("Graph canvas found")

            # Check for stats update (implies data loaded)
            print("Waiting for stats...")
            # Use specific check for node count > 0
            page.wait_for_function("document.getElementById('stat-nodes').innerText !== '0'", timeout=10000)
            print("Graph data loaded (nodes > 0)")

            # Check for live indicator
            page.wait_for_selector("#live-indicator", timeout=5000)
            print("Live indicator found")

        except Exception as e:
            print(f"Verification failed: {e}")
            # Take error screenshot
            page.screenshot(path=os.path.join(REPO_ROOT, "verification/system_knowledge_graph_error.png"))
            sys.exit(1)

        # Take success screenshot
        os.makedirs(os.path.join(REPO_ROOT, "verification"), exist_ok=True)
        screenshot_path = os.path.join(REPO_ROOT, "verification/system_knowledge_graph_v30.png")
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    # Ensure data exists
    if not os.path.exists(os.path.join(SERVE_DIR, "data/system_knowledge_graph.json")):
        print(f"Error: {os.path.join(SERVE_DIR, 'data/system_knowledge_graph.json')} not found. Run scripts/generate_system_graph.py first.")
        sys.exit(1)

    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give server a moment
    time.sleep(2)

    # Run verification
    try:
        verify_system_knowledge_graph()
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    sys.exit(0)
