from playwright.sync_api import sync_playwright
import os
import time
import http.server
import socketserver
import threading

PORT = 8001 # Changed port to avoid conflict if previous still running
DIRECTORY = "showcase"

def run_server():
    os.chdir(DIRECTORY)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()

def verify_ui():
    # Start server in thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2) # Wait for server

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Verify Neural Dashboard
        print("Verifying Neural Dashboard...")
        page.goto(f"http://localhost:{PORT}/neural_dashboard.html")
        page.wait_for_selector("#mynetwork") # Corrected ID
        time.sleep(3)
        page.screenshot(path="../verification/dashboard.png")

        # 2. Verify Deep Dive Reports
        print("Verifying Reports...")
        page.goto(f"http://localhost:{PORT}/deep_dive.html")
        page.wait_for_selector(".glass-panel")
        time.sleep(2)
        page.screenshot(path="../verification/reports.png")

        browser.close()

if __name__ == "__main__":
    verify_ui()
