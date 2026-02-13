import http.server
import socketserver
import threading
import time
import os
import sys
from playwright.sync_api import sync_playwright

PORT = 8082
Handler = http.server.SimpleHTTPRequestHandler

def start_server():
    # Only bind if port is free (retry logic or assume it's free/reuse addr)
    # SimpleHTTPRequestHandler serves from CWD
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("Serving at port", PORT)
            httpd.serve_forever()
    except OSError:
        print(f"Port {PORT} already in use, assuming server running.")

def verify_library():
    # Start server in thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root) # Ensure relative paths work for server

    url = f"http://localhost:{PORT}/showcase/credit_memo_automation.html"
    print(f"Verifying: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 800})
            page.goto(url)

            # 1. Verify Library Loads
            print("Checking Library List...")
            page.wait_for_selector("#library-list > div", timeout=10000)

            # 2. Check for Specific Companies
            library_text = page.inner_text("#library-list")
            print(f"Library Content: {library_text[:200]}...") # Print preview
            assert "TechCorp Inc." in library_text
            assert "Apple Inc." in library_text
            assert "Tesla Inc." in library_text
            assert "JPMorgan Chase" in library_text

            # 3. Click "Apple Inc." and verify Main Header
            print("Clicking Apple Inc. ...")
            # Using text locator for robustness
            page.click("text=Apple Inc.")

            print("Waiting for Memo Update...")
            # Wait for header to reflect Apple
            page.wait_for_selector("#memo-container h1", timeout=5000)

            # Allow time for JS render update
            time.sleep(1)

            header_text = page.inner_text("#memo-container h1")
            print(f"New Header: {header_text}")
            assert "Apple Inc." in header_text

            # 4. Take Screenshot of Apple Memo
            screenshot_dir = os.path.join(repo_root, "verification", "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            page.screenshot(path=os.path.join(screenshot_dir, "credit_memo_apple.png"))
            print("Saved credit_memo_apple.png")

            # 5. Click "Tesla Inc."
            print("Clicking Tesla Inc. ...")
            page.click("text=Tesla Inc.")
            time.sleep(1)
            header_text = page.inner_text("#memo-container h1")
            print(f"New Header: {header_text}")
            assert "Tesla Inc." in header_text

            page.screenshot(path=os.path.join(screenshot_dir, "credit_memo_tesla.png"))
            print("Saved credit_memo_tesla.png")

            print("VERIFICATION SUCCESS: Library Prototype Functional.")
            browser.close()

    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_library()
