import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Server configuration
PORT = 8125
SERVER_URL = f"http://localhost:{PORT}/showcase/system_brain.html"

def start_server():
    """Starts a simple HTTP server in a background thread."""
    root_dir = os.getcwd()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('localhost', PORT), Handler)
    httpd.serve_forever()

def verify_enhanced_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {SERVER_URL}")
        page.goto(SERVER_URL)

        # Verify Hardware Gauges
        try:
            page.wait_for_selector("#cpu-val", timeout=5000)
            cpu_val = page.locator("#cpu-val").inner_text()
            print(f"CPU Gauge Found: {cpu_val}")

            page.wait_for_selector("#gpu-val", timeout=5000)
            gpu_val = page.locator("#gpu-val").inner_text()
            print(f"GPU Gauge Found: {gpu_val}")

        except Exception as e:
            print(f"Timeout waiting for gauges: {e}")

        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/system_brain_enhanced.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    verify_enhanced_dashboard()
    sys.exit(0)
