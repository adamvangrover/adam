import os
import sys
import threading
import time
import http.server
import socketserver
from playwright.sync_api import sync_playwright

PORT = 8125

def start_server():
    os.chdir(".") # Ensure we are at root
    Handler = http.server.SimpleHTTPRequestHandler
    # Silence logs
    class SilentHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(("", PORT), SilentHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 1. Verify prompt_alpha.html
        url = f"http://localhost:{PORT}/showcase/apps/prompt_alpha.html"
        print(f"Navigating to {url}")
        page.goto(url)

        # Check title
        title = page.title()
        if "Prompt Alpha Terminal" not in title:
            raise AssertionError(f"Title mismatch: {title}")

        # Check Ticker
        page.wait_for_selector("#ticker-tape")

        # Check Feed List (should be populated by simulation)
        # Wait a bit for simulation to generate items
        page.wait_for_selector("#feed-list .feed-item", timeout=10000)

        # Check Chart
        page.wait_for_selector("#alpha-chart")

        page.screenshot(path="verification/prompt_alpha.png")
        print("Verified prompt_alpha.html")

        # 2. Verify analyst_os.html dock button
        url_os = f"http://localhost:{PORT}/showcase/analyst_os.html"
        print(f"Navigating to {url_os}")
        page.goto(url_os)

        # Check for dock button
        # Wait for dock to load
        page.wait_for_selector('.dock')

        dock_btn = page.locator('button[aria-label="Prompt Alpha"]')
        if dock_btn.count() == 0:
             raise AssertionError("Prompt Alpha dock button not found")

        print("Verified Prompt Alpha dock button in analyst_os.html")

        # 3. Verify index.html card
        url_index = f"http://localhost:{PORT}/showcase/index.html"
        print(f"Navigating to {url_index}")
        page.goto(url_index)

        # Check for card
        # Search for text "PROMPT ALPHA"
        if page.locator("text=PROMPT ALPHA").count() == 0:
            raise AssertionError("PROMPT ALPHA text not found in index.html")

        # Check link
        link = page.locator('a[href="apps/prompt_alpha.html"]')
        if link.count() == 0:
            raise AssertionError("Link to apps/prompt_alpha.html not found in index.html")

        print("Verified Prompt Alpha card in index.html")

        browser.close()

if __name__ == "__main__":
    # Check if port is in use, if so, pick another? No, just fail or hope 8125 is free.
    # Start server
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2) # Wait for server

    try:
        run_verification()
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)

    print("Verification passed!")
    sys.exit(0)
